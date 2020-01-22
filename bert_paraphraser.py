#! /usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import random
import logging
import random
import datetime

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss
import numpy as np
from tqdm import tqdm, trange
from transformers.tokenization_bert import BertTokenizer
from transformers.optimization import AdamW, WarmupLinearSchedule
from transformers import CONFIG_NAME, WEIGHTS_NAME
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from model import BertMouth
from data import make_dataloader

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument("--fix_word", default=None, type=str, required=True,
                        help="A fixed word in text generation.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available.")
    parser.add_argument('--seed',
                        type=int,
                        default=-1,
                        help="A random seed for initialization.")
    parser.add_argument("--samples", default=10, type=int,
                        help="The number of generated texts.")

    args = parser.parse_args()
    return args

def paraphrase(tokenizer, device, max_length=128,
             model=None, fix_word=None, samples=1):

    # モデルを読み込む、state_dictはパラメータを読み込む？
    model_state_dict = torch.load(os.path.join(model, "pytorch_model.bin"),
                                    map_location=device)
    model = BertMouth.from_pretrained(model,
                                        state_dict=model_state_dict,
                                        num_labels=tokenizer.vocab_size)
    # GPU/CPUにあわせる
    model.to(device)

    # 入力を分かち書き
    tokens = tokenizer.tokenize(fix_word)
    generated_token_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_type_id = [0] * max_length
    input_mask = [1] * len(generated_token_ids)
    while len(input_mask) < max_length:
        generated_token_ids.append(0)
        input_mask.append(0)

    generated_token_ids = torch.tensor([generated_token_ids],
                                        dtype=torch.long).to(device)
    input_type_id = torch.tensor(
        [input_type_id], dtype=torch.long).to(device)
    input_mask = torch.tensor([input_mask], dtype=torch.long).to(device)


    for j, _ in enumerate(tokens):
        # 文章のトークン１つをMASKに変換する
        # ヘッダは除くから、+1から
        generated_token_ids[0, j + 1] = tokenizer.vocab["[MASK]"]

        # 予測、(all_encoder_layers, _)が返り値
        logits = model(generated_token_ids, input_type_id, input_mask)[0]

        # MASKにした箇所で最も確率が高いトークンを取得し、置き換える
        sampled_token_id = torch.argmax(logits[j + 1])

        if sampled_token_id == generated_token_ids[0, j + 1]:
            logits[j + 1] = torch.min(logits[j + 1])
            sampled_token_id = torch.argmax(logits[j + 1])

        generated_token_ids[0, j + 1] = sampled_token_id

        # idから文字列に変換、結合
        sampled_sequence = [tokenizer.ids_to_tokens[token_id]
                            for token_id in generated_token_ids[0].cpu().numpy()]
        sampled_sequence = "".join(
            [
                token[2:] if token.startswith("##") else token
                for token in list(filter(lambda x: x != '[PAD]', sampled_sequence))
            ]
        )

        logger.info("sampled sequence: {}".format(sampled_sequence))
    


def main():
    # 引数を処理する
    args = parse_argument()

    # 乱数処理のシードをチェック
    if args.seed is not -1:
        # 各種の乱数処理のシードを固定化する
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # pytorchに利用するGPU/CPUを設定する
    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")

    if device != "cpu":
        # GPUの乱数シードを設定する
        torch.cuda.manual_seed_all(args.seed)

    # 事前学習済みのBERTモデルのTokernizerを読み込む
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=False,
                                              tokenize_chinese_chars=False)

    # 言い換え
    paraphrase(tokenizer, device, model=args.bert_model,
                fix_word=args.fix_word, samples=args.samples)


if __name__ == '__main__':
    main()
