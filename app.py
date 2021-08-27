#!/usr/bin/env python
# coding: utf-8
import warnings
import os
import argparse
import json
from collections import defaultdict
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from model import *
import clinical_eval
from clinical_joint import eval_joint
from clinical_eval import MhsEvaluator
from utils import *
from transformers import *
warnings.filterwarnings("ignore")

text = "術後両側肺野特に左肺優位にすりガラス影や網状影を認めます。前回と比べ概ね変化ありません。鈍化あり。前回と同様です。両肺尖部に胸膜肥厚あり。"
dir_file = open('./data/test.xml', 'w+', encoding='UTF-8')
dir_file.writelines(text)
dir_file.close()

#xml2conll
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
saved_model = "./ouall" 
model = torch.load(os.path.join(saved_model, 'model.pt'))
model = model.to(device)

xml_dir = "./data"
conll_dir = "./txt2conll"
doc_level = True
is_raw = False 
segmenter = 'mecab'
bert_dir = "./NICT_BERT-base_JapaneseWikipedia_32K_BPE"
 

bert_tokenizer = BertTokenizer.from_pretrained(
        bert_dir,
        do_lower_case=False,
        do_basic_tokenize=False,
        tokenize_chinese_chars=False
    )
bert_tokenizer.add_tokens(['[JASP]'])

train_scale = 1.0
with_dct = True
xml_list = [os.path.join(xml_dir, file) for file in sorted(os.listdir(xml_dir)) if file.endswith(".xml")]
print(f"total files: {len(xml_list)}")

if not os.path.exists(conll_dir):
    os.makedirs(conll_dir)

if not is_raw:
        batch_convert_document_to_conll(
            xml_list,
            os.path.join(
                conll_dir,
                f"single.conll"
            ),
            sent_tag=True,
            contains_modality=True,
            with_dct=with_dct,
            is_raw=is_raw,
            morph_analyzer_name=segmenter,
            bert_tokenizer=bert_tokenizer,
            is_document=doc_level
        )
else:
    for dir_file in xml_list:
            file_name = dir_file.split('/')[-1].rsplit('.', 1)[0]
            single_convert_document_to_conll(
                dir_file,
                os.path.join(
                    conll_dir,
                    f"{file_name}.conll"
                ),
                sent_tag=True,
                contains_modality=True,
                with_dct=with_dct,
                is_raw=is_raw,
                morph_analyzer_name=segmenter,
                bert_tokenizer=bert_tokenizer,
                is_document=doc_level
            )


test_file = './txt2conll/single.conll'
bert_max_len = 512

tokenizer = BertTokenizer.from_pretrained (
            saved_model,
            do_lower_case=True,
            do_basic_tokenize=False,
            tokenize_chinese_chars=False
        )
batch_size = 4

with open(os.path.join(saved_model, 'ner2ix.json')) as json_fi:
            bio2ix = json.load(json_fi)
with open(os.path.join(saved_model, 'mod2ix.json')) as json_fi:
            mod2ix = json.load(json_fi)
with open(os.path.join(saved_model, 'rel2ix.json')) as json_fi:
            rel2ix = json.load(json_fi)

test_output = './test_output/test.conll'
test_comments, test_toks, test_ners, test_mods, test_rels, _, _, _, _ = utils.extract_rel_data_from_mh_conll_v2(
                test_file,
                down_neg=0.0)
print(f"max sent len: {utils.max_sents_len(test_toks, tokenizer)}")
print(min([len(sent_rels) for sent_rels in test_rels]), max([len(sent_rels) for sent_rels in test_rels]))
print()

max_len = utils.max_sents_len(test_toks, tokenizer)
cls_max_len = max_len + 2

test_dataset, test_comment, test_tok, test_ner, test_mod, test_rel, test_spo = utils.convert_rels_to_mhs_v3(
                test_comments, test_toks, test_ners, test_mods, test_rels,
                tokenizer, bio2ix, mod2ix, rel2ix, cls_max_len, verbose=0)

cls_max_len = min(cls_max_len, bert_max_len)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

eval_joint(model, test_dataloader, test_comment, test_tok, test_ner, test_mod, test_rel, test_spo,
                       bio2ix, mod2ix, rel2ix, cls_max_len, device, "Final test dataset",
                       print_levels=(2, 2, 2), out_file=test_output, test_mode=False, verbose=0)
test_evaluator = MhsEvaluator(test_file, test_output)
test_evaluator.eval_ner(print_level=1)
test_evaluator.eval_mod(print_level=1)
# test_evaluator.eval_rel(print_level=2)
test_evaluator.eval_mention_rel(print_level=2)


#conll2xml
conll_dir = './test_output'
xml_result_dir = './xml_result'

conll_list = [os.path.join(conll_dir, file) for file in sorted(os.listdir(conll_dir)) if file.endswith(".conll")]
print(f"total files: {len(conll_list)}")
if not os.path.exists(xml_result_dir):
     os.makedirs(xml_result_dir)
for dir_conll in conll_list:
     file_name = dir_conll.split('/')[-1].rsplit('.', 1)[0]
     xml_out = os.path.join(xml_result_dir, f"{file_name}.xml")
     doc_conll = data_objects.MultiheadConll(dir_conll)
     doc_conll.doc_to_xml(xml_out)



