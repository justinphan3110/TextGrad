import argparse
import numpy as np
import torch
import pickle
import os
from src.logging.logger import LogModule
from src.attack.text_grad import PGDAttack

from src.substitution.bert_sub import BertSubstitutor
from src.substitution.roberta_sub import RobertaSubstitutor
from src.substitution.albert_sub import AlbertSubstitutor

from src.models.bert_model import BertVictimModel
from src.models.roberta_model import RoBERTaVictimModel
from src.models.albert_model import ALBERTVictimModel

from src.data_util.dataloader import load_attack_dataset, get_class_num, get_task_type

from src.attack.context import ctx_noparamgrad
from transformers import default_data_collator

import datasets
import time
from tqdm import tqdm

np.random.seed(107)

MODEL_CACHE_DIR = './model_cache/'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--norm', action = 'store_true')
parser.add_argument('--ste', action = 'store_true')
parser.add_argument('--rand', action = 'store_true')
parser.add_argument('--multi_sample', action = 'store_true')
parser.add_argument('--use_lm', action = 'store_true')
parser.add_argument('--cw', action = 'store_true')
parser.add_argument('--use_cache', action = 'store_true')

parser.add_argument('--eta_z', type = float, default = 0.8)
parser.add_argument('--eta_u', type = float, default = 0.8)
parser.add_argument('--iter_time', type = int, default = 20)
parser.add_argument('--sample_num', type = int, default = 20)
parser.add_argument('--final_sample', type = int, default = 20)
parser.add_argument('--modif', type = float, default = 0.25)
parser.add_argument('--size', type = int, default = 100)
parser.add_argument('--patience', type = int, default = 1)
parser.add_argument('--suffix', type = str, default = "")
parser.add_argument('--lm_beta', type = float, default = 0.1)

parser.add_argument('--visualize', action = 'store_true')


parser.add_argument("--victim", type = str, default = 'bert')
parser.add_argument("--dataset", type = str, default = 'sst')
parser.add_argument("--model_name_or_path", type = str, default = 'none')
parser.add_argument("--cache_dir", type = str, default = '')




succ = 0
fail = 0
skip = 0
oom = 0
failed_index_list = []

args = parser.parse_args()

if __name__ == '__main__':

    ## set device
    device = torch.device("cuda")

    ## Load Dataset
    attack_dataset = load_attack_dataset(args.dataset)
    sentence_pair = get_task_type(args.dataset)
    num_classes = get_class_num(args.dataset)


    ## Parse and Load Victim Model
    cache_dir = args.cache_dir
    if cache_dir == '':
        cache_dir = None
    print("loading model: ", args.model_name_or_path)
    print("loading checkpoints from ", cache_dir)
    if args.victim == 'bert':
        clsf = BertVictimModel(model_name_or_path = args.model_name_or_path, cache_dir = cache_dir, device = device, num_labels = num_classes,
                            max_len = 100)
    elif args.victim == 'roberta':
        clsf = RoBERTaVictimModel(model_name_or_path = args.model_name_or_path, cache_dir = cache_dir, device = device, num_labels = num_classes,
                            max_len = 100)
    elif args.victim == 'albert':
        clsf = ALBERTVictimModel(model_name_or_path = args.model_name_or_path, cache_dir = cache_dir, device = device, num_labels = num_classes,
                            max_len = 100)

    TEST_SIZE = args.size
    patience = args.patience 

    ## Parse and Get Substitution Method
    if args.victim == 'bert':
        substitutor = BertSubstitutor(model_type = 'bert-base-uncased', model_dir = MODEL_CACHE_DIR + 'bert_model/bert-base-uncased/masklm/',
                                    )
    elif args.victim == 'roberta':
        substitutor = RobertaSubstitutor(model_type = 'roberta-base', model_dir = MODEL_CACHE_DIR + 'roberta_model/roberta-base/masklm/',
                                        filter_words_file = './aux_files/vocab.txt',
                                        )
    elif args.victim == 'albert':
        substitutor = AlbertSubstitutor(model_type = 'albert-base-v2', model_dir = MODEL_CACHE_DIR + 'albert_model/albert-base-v2/masklm/',
                                            filter_words_file = './aux_files/vocab.txt'
                                        )

    ## Parse and Get Attack Model
    attacker = PGDAttack(victim_model = clsf, tokenizer = clsf.tokenizer, substitutor = substitutor, device = device, modification_rate = args.modif,
                        eta_z = args.eta_z, eta_u = args.eta_u, iter_time = args.iter_time, ste = args.ste, no_subword = True,
                        norm = args.norm, rand_init = args.rand,  multi_sample = args.multi_sample,
                        discrete_sample_num = args.sample_num, final_sample_time = args.final_sample,
                        use_lm = args.use_lm, lm_loss_beta= args.lm_beta,
                        use_cw_loss = args.cw, use_cache = args.use_cache,
                        victim = args.victim, num_classes = num_classes, sentence_pair = sentence_pair)
    attack_dataset = attack_dataset.select(range(1,TEST_SIZE))
    attack_dataset = attack_dataset.map(attacker.prepare_build_neighbor_matrix, batched=False, num_proc=1, desc="Building neighbor matrices")


    ## Load Log Module
    logger = LogModule()


    ## Conduct Attack
    count = 0
    index_count = 0
    modif_rate_list = []
    restartnum = 0
    corr = 0

    start_time = time.time()
    batch_size = 32
    # dataloader = torch.utils.data.DataLoader(attack_dataset, collate_fn=default_data_collator, batch_size=batch_size)

    for batch_idx in tqdm(range(0, len(attack_dataset), batch_size)):
    # for idx, batch in enumerate(tqdm(attack_dataset)):
        index_count += 1
        batch = attack_dataset[batch_idx:batch_idx + batch_size]
        sentences = batch['sentence']
        orig_labels = batch['label']
        
        # orig_score = clsf.predict([sentence])[0]
        # ## filter wrong samples
        # pred_label = np.argmax(orig_score)
        # if orig_label != pred_label:
        #     print("skipping wrong samples....")
        #     skip += 1
        #     count += 1
        #     continue
        # print("Attacking %d/%d sample.... "%(count, len(attack_dataset)))
        # count += batch_size
        # print(sentence)
        # try:
        t0 = time.time()
        with ctx_noparamgrad(clsf.model):
            # succ_examples, succ_pred_scores, succ_modif_rates, flag \
            #     = attacker.attack_preprocesed(batch, restart_num = patience)
            
            attack_logs = attacker.attack_preprocesed(batch, restart_num = patience)
        # print(succ_examples)
        print("Per Batch time: ", time.time() - t0 , " s")
        # assert False
        for sentence, label in zip(sentences, orig_labels):
            adv_exmaples, adv_pred_scores, adv_modif_rates, attack_flag = attack_logs[sentence]
            if not attack_flag:
                logger.record(False, batch_idx, sentence, adv_exmaples, label, -1)
                fail += 1
            else:
                succ += 1
                best_idx = torch.argmin(torch.stack(adv_modif_rates))
                adv_score = adv_pred_scores[best_idx]
                adv_label = np.argmax(adv_score)
                adv_example = adv_exmaples[best_idx]
                print("====="*5)
                print("sentence: ", sentence)
                print("adv_example: ",  adv_example)
                print("====="*5)
                logger.record(True, batch_idx, sentence, adv_exmaples, label, 1-label)
        
        print()
        print(f"[Succeeded / Failed / Skipped / OOM / Total] {succ} / {fail} / {skip} / {oom} / {index_count}")
        print()
    
    end_time = time.time()

    suffix = args.suffix
    if not os.path.exists("./attack_log/"):
        os.mkdir("./attack_log/")
    if suffix == "":
        save_dir = f'attack_log/textgrad_{args.dataset}_{args.victim}' + '.pkl'
    else:
        save_dir = f'attack_log/textgrad_{args.dataset}_{args.victim}_' + suffix + '.pkl'
    with open(save_dir,'wb') as f:
        pickle.dump(logger, f)
    
    print("time used: ", end_time - start_time)



