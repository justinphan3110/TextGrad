import numpy as np
import torch
import torch.nn.functional as F
from ..substitution.base_sub import BaseSubstitutor
from transformers import PreTrainedTokenizer
from ..models.base_model import BaseModel
from .attack_util import (match_subword_with_word, match_subword_with_word_albert, 
                            pos_tag, expand_with_match_index, match_subword_with_word_roberta,
                          STESample, STERandSelect, adjust_discrete_num)
import time
from typing import List

class PGDAttack():
    def __init__(self, victim_model: BaseModel, tokenizer: PreTrainedTokenizer, substitutor:BaseSubstitutor,
                    eta_z = 0.8, eta_u = 0.8, modification_rate = 0.25, iter_time = 20, max_neighbor_num = 50,
                    final_sample_time = 20, ste = True, norm = True, rand_init = True, no_subword = False, 
                    multi_sample = False, discrete_sample_num = 20, use_lm = False, lm_loss_beta = 0.1, use_cw_loss = True, 
                    device = torch.device("cuda"), num_classes = 2, victim = 'bert', use_cache = False, sentence_pair = False
                    ):
        self.victim_model = victim_model
        self.tokenizer = tokenizer
        self.substitutor = substitutor
        self.eta_z = eta_z
        self.eta_u = eta_u
        self.modification_rate = modification_rate

        self.iter_time = iter_time
        self.max_neighbor_num = max_neighbor_num
        self.final_sample_time = final_sample_time
        self.ste = ste
        self.norm = norm
        self.rand_init = rand_init
        self.no_subword = no_subword
        self.use_lm = use_lm

        self.multi_sample = multi_sample
        self.discrete_sample_num = discrete_sample_num

        self.lm_loss_beta = lm_loss_beta
        self.use_cw_loss = use_cw_loss
        self.use_cache = use_cache

        self.cw_tau = 0

        self.patience = 0
        self.device = device
        self.num_classes = num_classes
        self.victim = victim
        self.sentence_pair = sentence_pair
        self.input_embedding = self.victim_model.get_input_embedding()
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction = 'none')

    def tokenize_sentence(self, sentence1, sentence2 = None):
        '''
        sentence1: str
        sentence2: str
        '''
        # self.tokenizer.pad_token = self.tokenizer.sep_token
        result = self.tokenizer(text = sentence1, text_pair = sentence2, add_special_tokens = True, truncation = True ,padding='max_length', max_length=32)
        idx_list = result['input_ids']
        # print("idx_list", len(idx_list))
        attention_mask = result['attention_mask']
        token_type_ids = result['token_type_ids'] if 'token_type_ids' in result else None
        token_list = self.tokenizer.convert_ids_to_tokens(idx_list)
        assert len(token_list) == len(idx_list)
        output = (token_list, idx_list, attention_mask, token_type_ids)
        sentence1_tokens = token_list[1:-1]
        output += (sentence1_tokens, )
        return output

    def detokenize_idxs(self, idx_list):
        sentence = self.tokenizer.decode(idx_list, skip_special_tokens = True)
        return sentence
    
    def get_subtoken_mask(self, subword_list: List[str]):
        orig_mask = [1 for _ in subword_list]
        for idx in range(len(subword_list)):
            if subword_list[idx].startswith("##"):
                orig_mask[idx] = 0
                if idx > 0 and orig_mask[idx - 1] == 1:
                    orig_mask[idx - 1] = 0
        return orig_mask

    def build_neighbor_matrix(self, subword_list, match_index, pos_list):
        '''
        Build the substitution matrix for variable $z$ and $u$. The variable `mat` is a L*N matrix where L is the sequence length and N is the neighbor number.
        `mat` contains the substitution tokens for each token in the original sentence
        `site_mask` (a vector with L dimension) indicates which parts of tokens should be skipped during attacking, such as stopwords
        `sub_mask` is a L*N matrix indicating which part of substitution tokens should be skipped. For example, if a substituion token is stopwords, sub-words, etc, we 
        will not use it to replace the original token for attacking
        '''
        mat = torch.zeros([len(subword_list), self.max_neighbor_num], device = self.device, dtype = torch.long)
        subword_score_mat = torch.ones([len(subword_list), self.max_neighbor_num], device = self.device, dtype = torch.float32)
        origword_score_mat = torch.ones([len(subword_list), 1], device = self.device, dtype = torch.float32)
        site_mask = torch.ones([len(subword_list)], device = self.device, dtype = torch.long)

        if match_index is not None:
            ## As we have explained before, we match the POS of original words with its sub-words.
            expanded_pos_list = expand_with_match_index(pos_list, match_index)
            pos_mask = [1 if x in ['r','a','n','v'] else 0 for x in pos_list]
            expanded_pos_mask = expand_with_match_index(pos_mask, match_index)
            expanded_pos_mask = torch.tensor(expanded_pos_mask, device = self.device)
            site_mask *= expanded_pos_mask    ## mask those stopwords
        else:
            expanded_pos_list = ['none' for _ in range(len(subword_list))]

        t0 = time.time()
        ## the substitutor module will generate the substitution tokens as well as their MLM loss for regularization.
        substitute_for_sentence, lm_loss_for_sentence, orig_lm_loss_for_sentence = self.substitutor.get_neighbor_list(subword_list, site_mask, pos_list = expanded_pos_list)
        for i in range(mat.size(0)):
            if site_mask[i] == 0:
                continue
            substitute_words = substitute_for_sentence[i]
            lm_loss_for_word = lm_loss_for_sentence[i]
            orig_lm_loss = orig_lm_loss_for_sentence[i]
            if len(substitute_words) > self.max_neighbor_num:
                substitute_words = substitute_words[:self.max_neighbor_num]
                lm_loss_for_word = lm_loss_for_word[:self.max_neighbor_num]
            for j in range(len(substitute_words)):
                curr_subword = substitute_words[j]
                curr_subtokens = self.tokenizer.tokenize(curr_subword)
                substitute_token_id = self.tokenizer.convert_tokens_to_ids(curr_subtokens)
                if len(substitute_token_id) >= 2:
                    continue
                mat[i][j] = substitute_token_id[0]
                subword_score_mat[i][j] = lm_loss_for_word[j]
                origword_score_mat[i][0] = orig_lm_loss
        sub_mask = mat != 0

        zero_substitute_pos_mask = torch.sign(torch.sum(sub_mask, dim = 1))
        site_mask *= zero_substitute_pos_mask

        return mat, site_mask, sub_mask, subword_score_mat, origword_score_mat

    def build_sentence_pair_neighbor_matrix(self, subword_list, match_index, pos_list, sentence1_tokens, sentence2_tokens):
        '''
        see `build_neighbor_matrix` for explanation. The only difference is the first sentence will be all masked.
        '''
        mat = torch.zeros([len(subword_list), self.max_neighbor_num], device = self.device, dtype = torch.long)
        subword_score_mat = torch.ones([len(subword_list), self.max_neighbor_num], device = self.device, dtype = torch.float32)
        origword_score_mat = torch.ones([len(subword_list), 1], device = self.device, dtype = torch.float32)
        site_mask = torch.ones([len(subword_list)], device = self.device, dtype = torch.long)

        if match_index is not None:
            expanded_pos_list = expand_with_match_index(pos_list, match_index)
            pos_mask = [1 if x in ['r','a','n','v'] else 0 for x in pos_list]
            expanded_pos_mask = expand_with_match_index(pos_mask, match_index)
            expanded_pos_mask = torch.tensor(expanded_pos_mask, device = self.device)
            site_mask *= expanded_pos_mask
        else:
            expanded_pos_list = ['none' for _ in range(len(subword_list))]

        only_sen2_tokens = [self.tokenizer.cls_token] + sentence2_tokens + [self.tokenizer.sep_token]
        only_sen2_sitemask = site_mask[1 + len(sentence1_tokens):]
        only_sen2_poslist = expanded_pos_list[1 + len(sentence1_tokens):]
        assert len(only_sen2_tokens) == len(only_sen2_sitemask)
        substitute_for_sentence, lm_loss_for_sentence, orig_lm_loss_for_sentence = self.substitutor.get_neighbor_list(only_sen2_tokens, only_sen2_sitemask, pos_list = only_sen2_poslist)
        assert len(substitute_for_sentence) == len(only_sen2_tokens)

        for i in range(1 + len(sentence1_tokens), mat.size(0)):
            if site_mask[i] == 0:
                continue
            substitute_words = substitute_for_sentence[i - 1 - len(sentence1_tokens)]
            lm_loss_for_word = lm_loss_for_sentence[i - 1 - len(sentence1_tokens)]
            orig_lm_loss = orig_lm_loss_for_sentence[i - 1 - len(sentence1_tokens)]
            if len(substitute_words) > self.max_neighbor_num:
                substitute_words = substitute_words[:self.max_neighbor_num]
                lm_loss_for_word = lm_loss_for_word[:self.max_neighbor_num]
            for j in range(len(substitute_words)):
                curr_subword = substitute_words[j]
                curr_subtokens = self.tokenizer.tokenize(curr_subword)
                substitute_token_id = self.tokenizer.convert_tokens_to_ids(curr_subtokens)
                if len(substitute_token_id) >= 2 or len(substitute_token_id) <= 0:
                    continue
                mat[i][j] = substitute_token_id[0]
                subword_score_mat[i][j] = lm_loss_for_word[j]
                origword_score_mat[i][0] = orig_lm_loss
        sub_mask = mat != 0

        zero_substitute_pos_mask = torch.sign(torch.sum(sub_mask, dim = 1))
        site_mask *= zero_substitute_pos_mask

        return mat, site_mask, sub_mask, subword_score_mat, origword_score_mat

    def get_word_list(self, sentence):
        if type(sentence) == str:
            word_list = sentence.split()
        else:
            word_list = sentence
        return word_list
    
    def init_perturb_tensor(self, site_mask, sub_mask, eps=None, init_per_sample = 1, project = True):
        '''
        initialize the attack variable $z$ and $u$
        '''
        seq_len, neighbor_num = sub_mask.size()
        z_tensor = torch.ones([init_per_sample, seq_len], device = self.device, requires_grad = True, dtype = torch.double)
        u_tensor = torch.zeros([init_per_sample, seq_len, neighbor_num], device = self.device, dtype = torch.double).fill_(1/neighbor_num)

        if self.rand_init:
            torch.nn.init.uniform_(z_tensor, 0, 1)
            torch.nn.init.uniform_(u_tensor, 0, 1)

        site_mask_multi_init = site_mask.view(1, seq_len).repeat(init_per_sample, 1)
        sub_mask_multi_init = sub_mask.view(1, seq_len, neighbor_num).repeat(init_per_sample, 1, 1)
        z_tensor = (z_tensor * site_mask_multi_init).detach()
        u_tensor = (u_tensor * sub_mask_multi_init).detach()
        z_tensor = z_tensor.view(-1, seq_len)
        u_tensor = u_tensor.view(-1, seq_len, neighbor_num)
        if project:
            z_tensor = self.project_z_tensor(z_tensor, eps = eps).detach().clone().view(init_per_sample, seq_len)
            for i in range(init_per_sample):
                u_tensor[i] = self.project_u_tensor(u_tensor[i], site_mask = site_mask_multi_init[i], sub_mask = sub_mask_multi_init[i]).detach().clone().view(init_per_sample, seq_len, neighbor_num)

        z_tensor.requires_grad = True
        u_tensor.requires_grad = True

        return z_tensor, u_tensor
    
    def apply_perturb(self, z_tensor, u_tensor, site_mask, sub_mask, orig_embeddings, subword_embeddings, loss_incremental):
        '''
        Using current $z$ and $u$ to perturb the sentence. Will return the perturbed sentence in the embedding space.
        z_tensor / site_mask: (init_num, seq_len)
        u_tensor / sub_mask: (init_num, seq_len, neighbor_num)

        mask:      0:mask, will not be replaced  |   1:not mask, can be replaced

        orig_embeddings: (seq_len, hidden_dim)
        subword_embeddings: (seq_len, neighbor_num, hidden_dim)
        '''
        init_num, seq_len, neighbor_num = u_tensor.size()
        new_site_mask = site_mask.view(1, seq_len).repeat(init_num, 1)
        new_sub_mask = sub_mask.view(1, seq_len, neighbor_num).repeat(init_num, 1, 1)

        masked_z_tensor = z_tensor * new_site_mask
        masked_u_tensor = u_tensor * new_sub_mask
        
        if self.use_lm:
            loss_incremental = loss_incremental.view(1, seq_len, neighbor_num)
            subword_lm_loss = masked_z_tensor * torch.sum(masked_u_tensor * loss_incremental, dim = 2)     
        else:
            subword_lm_loss = None

        rand_thres = torch.rand(masked_z_tensor.size()).to(self.device)
        if self.ste:
            discrete_z = STESample.apply(masked_z_tensor - rand_thres)        
        else:
            discrete_z = masked_z_tensor        
        discrete_z = discrete_z.view(init_num, seq_len, 1)

        masked_u_tensor = masked_u_tensor.view(-1, neighbor_num)
        if self.ste:
            flat_site_mask = new_site_mask.view(-1).eq(1)
            masked_u_tensor = masked_u_tensor.view(-1, neighbor_num)
            masked_u_tensor = masked_u_tensor + 1e-8
            # print("masked_u_tensor[flat_site_mask]:", masked_u_tensor[flat_site_mask].shape)
            masked_u_tensor[flat_site_mask] = STERandSelect.apply(masked_u_tensor[flat_site_mask])
            discrete_u = masked_u_tensor.view(init_num, seq_len, neighbor_num)
        else:
            discrete_u = masked_u_tensor
        discrete_u = discrete_u.view(init_num, seq_len, neighbor_num, 1)

        orig_embeddings = orig_embeddings.view(1, seq_len, -1)
        subword_embeddings = subword_embeddings.view(1, seq_len, neighbor_num, -1)
        new_embeddings = (1 - discrete_z) * orig_embeddings + discrete_z * torch.sum(discrete_u * subword_embeddings, dim = 2)

        discrete_u = discrete_u.view(init_num, seq_len, neighbor_num)
        discrete_z = discrete_z.view(init_num, seq_len)

        return new_embeddings, discrete_z, discrete_u, subword_lm_loss

    def bisection(self, a, eps, xi = 1e-5, ub=1):
        '''
        bisection method to find the root for the projection operation of $z$
        '''
        pa = torch.clip(a, 0, ub)
        # print("torch.sum(pa).item()", torch.sum(pa).item())
        if torch.sum(pa).item() <= eps:
            upper_S_update = pa
        else:
            mu_l = torch.min(a-1)
            mu_u = torch.max(a)
            while torch.abs(mu_u - mu_l)>xi:

                mu_a = (mu_u + mu_l)/2
                gu = torch.sum(torch.clip(a-mu_a, 0, ub)) - eps
                gu_l = torch.sum(torch.clip(a-mu_l, 0, ub)) - eps
                if gu == 0: 
                    break
                if torch.sign(gu) == torch.sign(gu_l):
                    mu_l = mu_a
                else:
                    mu_u = mu_a
            upper_S_update = torch.clip(a-mu_a, 0, ub)
            
        return upper_S_update

    def project_z_tensor(self, z_tensor, eps):
        for i in range(z_tensor.size(0)):
            z_tensor[i] = self.bisection(z_tensor[i], eps)
            assert torch.sum(z_tensor[i]) <= eps + 1e-3,  f"{torch.sum(z_tensor[i]).item()}, {eps}"
        return z_tensor

    def norm_vector(self, vec):
        if torch.sum(vec) == 0:
            return vec
        norm_vec = vec / torch.norm(vec)
        return norm_vec
    
    def bisection_u(self, a, eps, xi = 1e-5, ub=1):
        '''
        bisection method to find the root for the projection operation of $u$
        '''
        pa = torch.clip(a, 0, ub)
        if np.abs(torch.sum(pa).item() - eps) <= xi:
            upper_S_update = pa
        else:
            mu_l = torch.min(a-1).item()
            mu_u = torch.max(a).item()
            while np.abs(mu_u - mu_l)>xi:
                mu_a = (mu_u + mu_l)/2
                gu = torch.sum(torch.clip(a-mu_a, 0, ub)) - eps
                gu_l = torch.sum(torch.clip(a-mu_l, 0, ub)) - eps + 1e-8
                gu_u = torch.sum(torch.clip(a-mu_u, 0, ub)) - eps
                if gu == 0: 
                    break
                elif gu_l == 0:
                    mu_a = mu_l
                    break
                elif gu_u == 0:
                    mu_a = mu_u
                    break
                if gu * gu_l < 0:  
                    mu_l = mu_l
                    mu_u = mu_a
                elif gu * gu_u < 0:  
                    mu_u = mu_u
                    mu_l = mu_a
                else:
                    print(a)
                    print(gu, gu_l, gu_u)
                    raise Exception()

            upper_S_update = torch.clip(a-mu_a, 0, ub)
        return upper_S_update    

    def project_u_tensor(self, u_tensor, site_mask, sub_mask):
        skip = site_mask == 0
        subword_opt = sub_mask != 0
        for i in range(u_tensor.size(0)):
            if skip[i]:
                continue
            u_tensor[i][subword_opt[i]] = self.bisection_u(u_tensor[i][subword_opt[i]], eps = 1)
            assert torch.abs(torch.sum(u_tensor[i][subword_opt[i]]) - 1) <= 1e-3
        return u_tensor


    def joint_optimize(self,z_tensor, u_tensor, z_grad, u_grad, site_mask, sub_mask, iter_time, eps):
        '''
        jointly optimize the two attack variables. The learning rate will decay with the attack iteration increases.
        '''
        z_update = self.eta_z / np.sqrt(iter_time) * z_grad
        u_update = self.eta_u / np.sqrt(iter_time) * u_grad
        z_tensor_update = z_tensor + z_update
        u_tensor_update = u_tensor + u_update

        z_tensor_list = []
        u_tensor_list = []
        for i in range(z_tensor_update.size(0)):
            z_tensor_res = self.bisection(z_tensor_update[i], eps = eps[i],)
            z_tensor_list.append(z_tensor_res)

        # t1 = time.time()
        for i in range(u_tensor_update.size(0)):
            u_tensor_res = self.project_u_tensor(u_tensor_update[i], site_mask[i], sub_mask[i])
            u_tensor_list.append(u_tensor_res)

        z_tensor_res = torch.stack(z_tensor_list, dim = 0)
        u_tensor_res = torch.stack(u_tensor_list, dim = 0)
        return z_tensor_res, u_tensor_res
    
    def joint_optimize_batch(self, z_tensor, u_tensor, z_grad, u_grad, site_mask, sub_mask, iter_time, eps):
        '''
        jointly optimize the two attack variables. The learning rate will decay with the attack iteration increases.
        '''
        z_update = self.eta_z / np.sqrt(iter_time) * z_grad
        u_update = self.eta_u / np.sqrt(iter_time) * u_grad
        z_tensor_update = z_tensor + z_update
        u_tensor_update = u_tensor + u_update
        z_tensor_list = []
        u_tensor_list = []
        # t0 = time.time()
        # print("z_tensor_update.shape", z_tensor_update.shape)
        # print("u_tensor_update.shape", u_tensor_update.shape)

        z_tensor_res = self.bisection_batch(z_tensor_update, eps=eps)

        t0 = time.time()
        for i in range(u_tensor_update.size(0)):
            u_tensor_res = self.project_u_tensor(u_tensor_update[i], site_mask[i], sub_mask[i])
            u_tensor_list.append(u_tensor_res)
        u_tensor_res = torch.stack(u_tensor_list, dim = 0)
        return z_tensor_res, u_tensor_res
    
    
    def bisection_batch(self, a, eps, xi=1e-5, ub=1):
        '''
        bisection method to find the root for the projection operation of $z$
        '''
        batch_size = a.size(0)
        pa = torch.clip(a, 0, ub)

        # Create a mask where the sum along each row is less than or equal to eps
        sum_mask = (torch.sum(pa, dim=1) <= eps).unsqueeze(1)

        # Calculate initial mu_l and mu_u for each row
        mu_l = torch.min(a - 1, dim=1).values.unsqueeze(1)
        mu_u = torch.max(a, dim=1).values.unsqueeze(1)

        while torch.abs(mu_u - mu_l).max() > xi:
            mu_a = (mu_u + mu_l) / 2
            gu = torch.sum(torch.clip(a - mu_a, 0, ub), dim=1) - eps
            gu_l = torch.sum(torch.clip(a - mu_l, 0, ub), dim=1) - eps

            same_sign_mask = (torch.sign(gu) == torch.sign(gu_l)).unsqueeze(1)
            mu_l = torch.where(same_sign_mask, mu_a, mu_l)
            mu_u = torch.where(~same_sign_mask, mu_a, mu_u)

        upper_S_update = torch.clip(a - mu_a, 0, ub)

        # Combine the results of pa and upper_S_update using sum_mask
        result = torch.where(sum_mask, pa, upper_S_update)
        return result

    def discretize_z(self, z_tensor, site_mask = None):
        z_tensor[site_mask != 1] = -10000
        rand_thres = torch.tensor(np.random.uniform(size=z_tensor.size())).to(self.device)
        discrete_z = torch.where(z_tensor > rand_thres, 1, 0)
        return discrete_z

    def discretize_u(self, u_tensor, site_mask = None):
        res = []    
        for i in range(u_tensor.size(0)):
            if site_mask[i] == 0:
                res.append(-1)
                continue
            prob = u_tensor[i].cpu().detach().numpy()
            prob = prob / np.sum(prob)
            substitute_idx = np.random.choice(u_tensor.size(1), p = prob)
            res.append(substitute_idx)
        return torch.tensor(res, device = self.device)

    def apply_substitution(self, discrete_z, discrete_u, idx_list, subword_idx_mat):
        discrete_z = np.reshape(discrete_z, newshape = [-1])
        discrete_u = np.reshape(discrete_u, newshape = [-1])
        replace_position = np.where(discrete_z == 1)[0]
        substitute_idx = discrete_u
        new_word_idx_list = idx_list[:]
        if np.sum(discrete_z) != 0:
            for i in range(len(replace_position)):
                curr_pos = replace_position[i]
                curr_subword_idx = subword_idx_mat[curr_pos][substitute_idx[curr_pos]]
                new_word_idx_list[curr_pos] = curr_subword_idx
        
        if not self.sentence_pair:
            sentence = self.detokenize_idxs(new_word_idx_list[1:-1])
        else:
            sep_index = -1
            for i in range(len(new_word_idx_list)):
                if new_word_idx_list[i] == self.tokenizer.sep_token_id:
                    sep_index = i
                    break
            assert sep_index > 0
            sentence = self.detokenize_idxs(new_word_idx_list[sep_index + 1:-1])  ## remove [CLS]/[SEP]
        return sentence
    

# =================================================================================================================================================================================================================================
# =================================================================================================================================================================================================================================
# =================================================================================================================================================================================================================================
    def perturb_batch(self, idx_list, orig_label, site_mask, sub_mask, subword_idx_mat, loss_incremental, 
                      attention_mask, token_type_ids, attack_word_num, eps):
        
        # print("subword_idx_mat.shape", subword_idx_mat.shape)
        # print("idx_list.shape", idx_list.shape)
        # print("orig_label.shape", orig_label.shape)
        # print("site_mask.shape", site_mask.shape)
        # print("sub_mask.shape", sub_mask.shape)
        # print("subword_idx_mat.shape", subword_idx_mat.shape)
        # print("loss_incremental.shape", loss_incremental.shape)

        # print("subword_idx_mat.shape", subword_idx_mat.shape)
        batch_size, seq_len, neighbor_num = subword_idx_mat.size()
        local_discrete_num = self.discrete_sample_num
        # local_discrete_num = adjust_discrete_num(self.victim, seq_len, self.discrete_sample_num)

        idx_tensor = [torch.LongTensor(l).to(self.device) for l in idx_list]
        # attention_mask = torch.tensor(attention_mask).to(self.device)
        # attention_mask = torch.unsqueeze(attention_mask, 0).float()
        if token_type_ids is not None:
            token_type_ids = torch.tensor(token_type_ids).to(self.device)
            token_type_ids = torch.unsqueeze(token_type_ids, 0)

        # print("site_mask", site_mask.shape) 
        # print("sub_mask", sub_mask.shape)

        b_z_tensor = []
        b_u_tensor = []
        b_labels = []
        b_succ_discrete_z_list = []
        b_succ_discrete_u_list = []
        b_orig_embeddings = []
        b_subword_embeddings = []
        for b_idx in range(batch_size):
            zt, ut = self.init_perturb_tensor(site_mask[b_idx], sub_mask[b_idx], eps=eps[b_idx], project = True, init_per_sample = 1) #Note later change site_mask and sub_mask to list of bs[i]
            b_z_tensor.append(zt)
            b_u_tensor.append(ut)
            b_labels.append(torch.LongTensor([orig_label[b_idx]]).to(self.device))
            b_succ_discrete_z_list.append([])
            b_succ_discrete_u_list.append([])

            b_orig_embeddings.append(self.input_embedding(idx_tensor[b_idx]))
            b_subword_embeddings.append(self.input_embedding(subword_idx_mat[b_idx]))

    
        # labels = torch.LongTensor([orig_label]).to(self.device)        
        t0 = time.time()
        joint_optimize_time = 0
        forward_loop_time = 0
        backward_time = 0
        discrete_time = 0

        for i in range(self.iter_time + 1):

            b_expanded_labels = [[] for _ in range(batch_size)]
            b_expanded_attention_mask = [[] for _ in range(batch_size)]
            b_new_embeddings_list = [[] for _ in range(batch_size)]
            b_discrete_z_list = [[] for _ in range(batch_size)]
            b_discrete_u_list = [[] for _ in range(batch_size)]
            b_subword_lm_loss_list = [[] for _ in range(batch_size)]
            for b_idx in range(batch_size):
                b_expanded_labels[b_idx].extend(b_labels[b_idx].repeat(local_discrete_num))
                b_expanded_attention_mask[b_idx].extend(attention_mask[b_idx].repeat(local_discrete_num, 1))
                for _ in range(local_discrete_num):
                    # print("loss_incremental.shape", loss_incremental.shape)
                    new_embeddings, discrete_z, discrete_u, subword_lm_loss = self.apply_perturb(b_z_tensor[b_idx], b_u_tensor[b_idx], site_mask[b_idx], sub_mask[b_idx], b_orig_embeddings[b_idx], b_subword_embeddings[b_idx], loss_incremental[b_idx])
                    b_discrete_z_list[b_idx].extend(discrete_z)
                    b_discrete_u_list[b_idx].extend(discrete_u)
                    b_new_embeddings_list[b_idx].extend(new_embeddings)      
                    b_subword_lm_loss_list[b_idx].extend(subword_lm_loss)

            if self.use_lm:
                b_subword_lm_loss_list = torch.stack([torch.stack(t, dim=0) for t in b_subword_lm_loss_list], dim=0)
            
            b_expanded_labels = torch.stack([torch.stack(t, dim=0) for t in b_expanded_labels], dim=0)
            b_expanded_attention_mask = torch.stack([torch.stack(t, dim=0) for t in b_expanded_attention_mask], dim=0)
            b_new_embeddings_list = torch.stack([torch.stack(t, dim=0) for t in b_new_embeddings_list], dim=0)
            b_discrete_z_list = torch.stack([torch.stack(t, dim=0) for t in b_discrete_z_list], dim=0)
            b_discrete_u_list = torch.stack([torch.stack(t, dim=0) for t in b_discrete_u_list], dim=0)

            t1 = time.time()
            result = self.victim_model.predict_via_embedding(b_new_embeddings_list.reshape(local_discrete_num*batch_size, seq_len, -1).float()\
                                                             , b_expanded_attention_mask.reshape(local_discrete_num*batch_size, seq_len) \
                                                             , None)
            forward_loop_time += time.time() - t1
            logits = result.logits   ##  batch_size*local_discrete_num, num_classes
            
            print("logits.shape", logits.shape, "b_new_embeddings_list.shape", b_new_embeddings_list.shape)



            scores = F.softmax(logits, dim = -1)
            scores = scores.reshape(batch_size, local_discrete_num, -1)
            # print("scores.shape", scores.shape)
            logit_mask = F.one_hot(b_expanded_labels.view(local_discrete_num*batch_size), num_classes = self.num_classes)
            # print("logit_mask.shape", logit_mask.shape)
            logit_orig = torch.sum(logits * logit_mask, axis = -1)
            # print("logit_orig.shape", logit_orig.shape)
            logit_others, _ = torch.max(logits - 99999 * logit_mask, axis = -1)
            cw_loss = F.relu(logit_orig - logit_others + self.cw_tau)
            loss = cw_loss
            loss_values = -cw_loss
            loss_values = loss_values.reshape(batch_size, local_discrete_num)
            # print("loss_values.shape ", loss_values.shape)
            target_indices = torch.argmax(loss_values, dim=1)
            # print("target index", target_indices)
            worst_scores = scores[range(batch_size), target_indices]
            # print("worst_score.shape", worst_scores.shape)

            if self.use_lm:
                mean_loss_values = torch.mean(loss_values, dim=1)
                mean_subword_lm_loss_list = torch.mean(b_subword_lm_loss_list, dim=(1, 2))
                loss = torch.mean(mean_loss_values + self.lm_loss_beta * mean_subword_lm_loss_list)
                # loss = torch.mean(loss_values) + self.lm_loss_beta * torch.mean(subword_lm_loss_list)
            else:
                loss = torch.mean(loss_values)
                assert False, "Not implemented for non use_lm loss yet: TODO"
            # print("Loss ", loss)
            t1 = time.time()
            loss.backward(retain_graph = True)
            backward_time += time.time() - t1

            t2 = time.time()
            z_grads = []
            u_grads = []
            for b_idx in range(batch_size):
                target_index = target_indices[b_idx]
                discrete_z = b_discrete_z_list[b_idx][target_index]
                discrete_u = b_discrete_u_list[b_idx][target_index]
                z_grad = b_z_tensor[b_idx].grad
                u_grad = b_u_tensor[b_idx].grad
                
                z_grads.append(z_grad)
                u_grads.append(u_grad)
                
                curr_model_prediction = worst_scores[b_idx]                          
                z_grad = self.norm_vector(z_grad)
                for idx in range(len(site_mask[b_idx])):
                    if site_mask[b_idx][idx] == 1:
                        u_grad[0][idx] = self.norm_vector(u_grad[0][idx])

                if torch.argmax(curr_model_prediction) != orig_label and self.ste:
                    curr_discrete_z = discrete_z.detach().clone().view(-1).cpu().numpy()
                    curr_discrete_u = torch.argmax(discrete_u.detach().clone(), dim = -1).view(-1).cpu().numpy()
                    b_succ_discrete_z_list[b_idx].append(curr_discrete_z)
                    b_succ_discrete_u_list[b_idx].append(curr_discrete_u)

                t1 = time.time()
            discrete_time += time.time() - t2

            t1 = time.time()
            z_tensor_opt, u_tensor_opt = self.joint_optimize_batch(torch.stack(b_z_tensor).squeeze(), \
                                                                   torch.stack(b_u_tensor).squeeze(), \
                                                                    torch.stack(z_grads).detach().clone().squeeze(dim=1), \
                                                                    torch.stack(u_grads).detach().clone().squeeze(dim=1), site_mask, \
                                                                    sub_mask, i + 1, \
                                                                    torch.stack([torch.tensor(e) for e in eps]).to(self.device))
            # assert False
            # print("z_tensor_opt[b_idx].unsqueeze(0)", z_tensor_opt[0].unsqueeze(0).shape)
            # print("u_tensor_opt[b_idx].unsqueeze(0)", u_tensor_opt[0].unsqueeze(0).shape)
            # print("u_tensor_opt.shape", u_tensor_opt.shape)
            for b_idx in range(batch_size):
                # print(f"u_tensor_opt[{b_idx}]", u_tensor_opt[b_idx])
                b_z_tensor[b_idx].data = z_tensor_opt[b_idx].unsqueeze(0)
                b_u_tensor[b_idx].data = u_tensor_opt[b_idx].unsqueeze(0)
                b_z_tensor[b_idx].grad.zero_()
                b_u_tensor[b_idx].grad.zero_()
            # assert False
            joint_optimize_time += time.time() - t1
            # print("b_z_tensor[b_idx].shape", b_z_tensor[b_idx].shape)
            # print("z_tensor_opt.shape", z_tensor_opt.shape)
            # print("u_tensor_opt.shape", u_tensor_opt.shape)
            # print("len(b_z_tensor)", len(b_z_tensor))
            # print("b_z_tensor[0].shape", b_z_tensor[0].shape)
            # assert False
            
        
        # print("backward_time time: ", backward_time, "s")
        # print("joint_optimize_time: ", joint_optimize_time, "s")
        # print("forward_loop_time: ", forward_loop_time, "s")
        # print("discrete_time: ", discrete_time, "s")
        # print("pertub_loop time: ", time.time() - t0, "s")
        # print("===="*10)

        return_samples = [] # [ [succ_example], [succ_pred_score] , [succ_modif_rate], [flag] ] 
        for b_idx in range(batch_size):
            succ_discrete_z_list = b_succ_discrete_z_list[b_idx]
            succ_discrete_u_list = b_succ_discrete_u_list[b_idx]
            z_tensor = b_z_tensor[b_idx]
            u_tensor = b_u_tensor[b_idx]

            # TODO: need to make this to [b_idx]
            # subword_idx_mat = subword_idx_mat
            # site_mask = site_mask
            # orig_label = orig_label
            # idx_list = idx_list
            return_samples.append(self.extract_adv_samples(succ_discrete_z_list=succ_discrete_z_list, \
                                                           succ_discrete_u_list=succ_discrete_u_list, \
                                                            attack_word_num=attack_word_num[b_idx], 
                                                            idx_list=idx_list[b_idx], \
                                                            z_tensor=z_tensor, \
                                                            u_tensor=u_tensor, \
                                                            subword_idx_mat=subword_idx_mat[b_idx], \
                                                            site_mask=site_mask[b_idx], \
                                                            orig_label=orig_label[b_idx]
                                                        )
                                )
        return return_samples

    def extract_adv_samples(self, \
                            succ_discrete_z_list, \
                            succ_discrete_u_list, \
                            attack_word_num, \
                            idx_list, \
                            z_tensor, u_tensor, \
                            subword_idx_mat, site_mask, orig_label):
        succ_examples = []
        succ_pred_scores = []
        modif_rates = []
        adv_sentence_list = []
        if self.use_cache:
            for i in range(len(succ_discrete_z_list)):
                discrete_z = succ_discrete_z_list[i]
                discrete_u = succ_discrete_u_list[i]
                modification_rate = np.sum(discrete_z == 1) / attack_word_num
                if modification_rate > self.modification_rate:
                    continue
                modif_rates.append(modification_rate)
                adv_sentence = self.apply_substitution(discrete_z, discrete_u, idx_list, subword_idx_mat.detach().cpu().numpy())
                adv_sentence_list.append(adv_sentence)
        detached_z_tensor = z_tensor.detach().clone()[0]
        detached_u_tensor = u_tensor.detach().clone()[0]
        for i in range(self.final_sample_time):

            discrete_z = self.discretize_z(detached_z_tensor, site_mask = site_mask).detach().cpu().numpy()
            discrete_u = self.discretize_u(detached_u_tensor, site_mask = site_mask).detach().cpu().numpy()


            modification_rate = np.sum(discrete_z == 1) / attack_word_num
            if modification_rate > self.modification_rate:
                continue
            adv_sentence = self.apply_substitution(discrete_z, discrete_u, idx_list, subword_idx_mat.detach().cpu().numpy())
            adv_sentence_list.append(adv_sentence)
            modif_rates.append(modification_rate)

        if len(adv_sentence_list) == 0:
            return [],[],[], False
        
        pred_prob = self.victim_model.predict(adv_sentence_list)
        orig_label_score = pred_prob[:, orig_label]
        
        pred_label = np.argmax(pred_prob, axis = -1)
        succ_idxs = np.where(pred_label != orig_label)[0]
        
        if len(succ_idxs) > 0:
            succ_examples = [adv_sentence_list[x] for x in succ_idxs]
            succ_pred_scores = pred_prob[succ_idxs]
            succ_modif_rates = [modif_rates[x] for x in succ_idxs]
            return succ_examples, succ_pred_scores, succ_modif_rates, True
        else:
            best_perturb = np.argmin(orig_label_score)

            return [adv_sentence_list[best_perturb]], [],[], False

    def prepare_build_neighbor_matrix(self, sample):
        sentence = sample["sentence"]
        label = sample["label"]

        tokens, idx_list, attention_mask, token_type_ids, sentence1_tokens = self.tokenize_sentence(sentence)
        sentence_tr = self.tokenizer.convert_tokens_to_string(sentence1_tokens)
        word_list = sentence_tr.split()
        pos_list = ['none'] + pos_tag(word_list) + ['none']
        word_list = [self.tokenizer.cls_token] + word_list + [self.tokenizer.sep_token]
        attack_word_num = len(word_list[1:-1])
        eps = int(self.modification_rate * len(word_list[1:-1]))
        
        try:
            ## What is these lines of codes doing? Since we hope to skip those stopwords and only perturb nouns, verbs, adjectives, and adverbs,
            ## we need to first pos-tagging the input sentence. However, the pos-tagging is word-level instead of token-level. Regarding on those
            ## words that are tokenized into sub-words by the langauge model's tokenizer, we need to assign the POS of the original word to these
            ## sub-words. Therefore, we need to find the mapping between the original word and the subwords using the `match_subword_with_word` function

            ## some words cannot be tokenized and will cause errors when matching subwords with original words
            ## one example from RTE dataset is "CAMDEN, N.J. (Reuters) — Three Muslim brothers from Albania ...". 
            ## The ALBERT tokenizer will tokenize "—" into '⁇' and the following matching algorithm will fail.
            ## In that case, we will delete the Part-of-speech(POS) constraint for that example, and ignore the POS when attacking
            if self.victim == 'bert':
                match_index = match_subword_with_word(tokens, word_list)
            elif self.victim == 'roberta':
                match_index = match_subword_with_word_roberta(tokens, word_list, self.tokenizer)
            elif self.victim == 'albert':
                match_index = match_subword_with_word_albert(tokens, word_list)
        except:
            match_index = None
        subword_idx_mat, site_mask, sub_mask, subword_score_mat, origword_score_mat = self.build_neighbor_matrix(tokens, match_index, pos_list)
        return {
            "subword_idx_mat": subword_idx_mat,
            "site_mask": site_mask,
            "sub_mask": sub_mask,
            "subword_score_mat": subword_score_mat,
            "origword_score_mat": origword_score_mat,
            "label": label,
            "idx_list": idx_list,
            "attention_mask": attention_mask, 
            "token_type_ids": token_type_ids,
            "attack_word_num": attack_word_num,
            "eps": eps,
            "sentence": sentence,
            "tokens": tokens,
        }

    def attack_preprocesed(self, sample, restart_num = 10):
        # print("sample" , len(sample["subword_idx_mat"]))
        batch_size = len(sample["subword_idx_mat"])
        subword_idx_mat = torch.stack([torch.tensor(s).to(self.device) for s in sample["subword_idx_mat"]])
        site_mask = torch.stack([torch.tensor(s).to(self.device) for s in sample["site_mask"]])
        sub_mask = torch.stack([torch.tensor(s).to(self.device) for s in sample["sub_mask"]])
        subword_score_mat = torch.stack([torch.tensor(s).to(self.device) for s in sample["subword_score_mat"]])
        origword_score_mat = torch.stack([torch.tensor(s).to(self.device) for s in sample["subword_score_mat"]])
        attention_mask = torch.stack([torch.tensor(s).to(self.device) for s in sample["attention_mask"]])
        attack_word_num = torch.stack([torch.tensor(s).to(self.device) for s in sample["attack_word_num"]])

        # subword_idx_mat = torch.tensor(sample["subword_idx_mat"]).to(self.device)
        # site_mask = torch.tensor(sample["site_mask"]).to(self.device)
        # sub_mask = torch.tensor(sample["sub_mask"]).to(self.device)
        # subword_score_mat = torch.tensor(sample["subword_score_mat"]).to(self.device)
        # origword_score_mat = torch.tensor(sample["subword_score_mat"]).to(self.device)
        # attention_mask = torch.tensor(sample["attention_mask"]).to(self.device)
        # attack_word_num = torch.tensor(sample["attack_word_num"]).to(self.device)
        token_type_ids = None
        orig_label = sample["label"]
        idx_list = sample["idx_list"]
        eps = sample["eps"]
        sentence = sample["sentence"]
        tokens = sample["tokens"]

        # self.eps = eps
        # if eps < 1:
        #     return [],[],[], False
        
        if self.use_lm:
            # TODO: Maybe this lm_loss is wrong
            lm_loss = torch.log(subword_score_mat) - torch.log(origword_score_mat)
        else:
            lm_loss = None

        attacked_logs = {}
        for patience in range(restart_num):
            if self.no_subword and patience <= 1:
                ## Most pre-trained language models use byte-pair encoding. Since TextGrad perturb the sentence in token-level, it is possible that
                ## some subwords are perturbed. For example, "surprising" could be tokenized into "surpris" and "ing". Perturbing such subwords simutanesouly
                ## could lead to some grammatical errors. To relieve this problem, we mask subwords and do not perturb them in several attack trials 
                ## If the attack cannot succeed without keeping subwords unchange, then we ignore the subword constraints and attack all tokens 
                new_site_mask = []
                for b_idx in range(batch_size):
                    subtoken_mask = self.get_subtoken_mask(tokens[b_idx])
                    subtoken_mask = torch.tensor(subtoken_mask, device = self.device)
                    new_site_mask.append(site_mask[b_idx] * subtoken_mask)
                new_site_mask = torch.stack(new_site_mask)
            else:
                # new_site_mask = [s.detach().clone() for s in site_mask]
                # new_sub_mask = [s.detach().clone() for s in sub_mask]
                new_site_mask = site_mask.detach().clone()
            new_sub_mask = sub_mask.detach().clone()
            perturbed_samples = self.perturb_batch(idx_list, orig_label, new_site_mask, new_sub_mask, subword_idx_mat, \
                                                lm_loss, attention_mask, token_type_ids, attack_word_num, eps)
            successful_attacks = set()
            for b_idx in range(len(perturbed_samples)):
                adv_exmaples, adv_pred_scores, adv_modif_rates, attack_flag = perturbed_samples[b_idx]
                attacked_logs.setdefault(sentence[b_idx] , (adv_exmaples, adv_pred_scores, adv_modif_rates, attack_flag))
                if attack_flag:
                    transformed_advs = []
                    for adv in adv_exmaples:
                        if sentence[0].isupper() and adv[0].islower():
                            recovered_str = adv[0].upper() + adv[1:]
                            transformed_advs.append(recovered_str)
                        else:
                            transformed_advs.append(adv)
                    adv_exmaples = transformed_advs
                    attacked_logs[sentence[b_idx]] = (adv_exmaples, adv_pred_scores, adv_modif_rates, attack_flag)
                    successful_attacks.add(b_idx)

            # return if all sucess
            if len(successful_attacks) == batch_size:
                return attacked_logs
            # Re-Sample batch for unsucessfull samples
            subword_idx_mat = torch.stack([t for i, t in enumerate(subword_idx_mat) if i not in successful_attacks])
            site_mask = torch.stack([t for i, t in enumerate(site_mask) if i not in successful_attacks])
            sub_mask = torch.stack([t for i, t in enumerate(sub_mask) if i not in successful_attacks])
            subword_score_mat = torch.stack([t for i, t in enumerate(subword_score_mat) if i not in successful_attacks])
            origword_score_mat = torch.stack([t for i, t in enumerate(origword_score_mat) if i not in successful_attacks])
            attention_mask = torch.stack([t for i, t in enumerate(attention_mask) if i not in successful_attacks])
            attack_word_num = torch.stack([t for i, t in enumerate(attack_word_num) if i not in successful_attacks])

            orig_label = [l for i, l in enumerate(orig_label) if i not in successful_attacks]
            idx_list = [l for i, l in enumerate(idx_list) if i not in successful_attacks]
            eps = [l for i, l in enumerate(eps) if i not in successful_attacks]
            sentence = [l for i, l in enumerate(sentence) if i not in successful_attacks]
            tokens = [l for i, l in enumerate(tokens) if i not in successful_attacks]
            batch_size = len(subword_idx_mat)





        # return adv_exmaples, adv_pred_scores, adv_modif_rates, attack_flag
        return attacked_logs

