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

    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir", default="./models", type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--train_file", default=None,
                        type=str, help="A file path for training.")
    parser.add_argument("--valid_file", default=None,
                        type=str, help="A file path for validation.")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_generate", action='store_true',
                        help="Whether to generate text.")
    parser.add_argument("--train_batch_size", default=32,
                        type=int, help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=5e-5,
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available.")
    parser.add_argument('--seed',
                        type=int,
                        default=-1,
                        help="A random seed for initialization.")
    parser.add_argument('--max_iter',
                        type=int,
                        default=10,
                        help="The number of iterations in text generation.")
    parser.add_argument('--seq_length',
                        type=int,
                        default=50,
                        help="The sequence length generated.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--fix_word", default=None, type=str,
                        help="A fixed word in text generation.")
    parser.add_argument("--samples", default=10, type=int,
                        help="The number of generated texts.")

    args = parser.parse_args()
    return args


def save(args, model, tokenizer, name):
    output_dir = os.path.join(args.output_dir, name)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)

    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)

    tokenizer.save_vocabulary(output_dir)

    output_args_file = os.path.join(output_dir, 'training_args.bin')
    torch.save(args, output_args_file)

    logger.info("Model are saved in {}".format(output_dir))


def train(args, tokenizer, device):
    logger.info("loading data")

    # 教師データを読み込む
    train_dataloader = make_dataloader(args.train_file, args.max_seq_length,
                                       args.train_batch_size, tokenizer)
    # 評価データを読み込む
    valid_dataloader = make_dataloader(args.valid_file, args.max_seq_length,
                                       args.train_batch_size, tokenizer)

    logger.info("building model")

    # BertMouthモデルの事前学習モデル（おそらく通常のBERTモデルと同じ構造）を読み込む
    model = BertMouth.from_pretrained(args.bert_model,
                                      num_labels=tokenizer.vocab_size)
    # GPU/CPUを設定する
    model.to(device)

    # 名前がpoolerではないパラメータを取得する
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    logger.info("setting optimizer")

    # decayに含まれるパラメータとそうでないパラメータにわける
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # ステップは教師データの大きさ * epoch
    optimization_steps = len(train_dataloader) * args.num_train_epochs
    # 最適化アルゴリズムの指定
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    # スケジューラーは学習率を調整してくれる
    scheduler = WarmupLinearSchedule(optimizer,
                                     warmup_steps=0,
                                     t_total=optimization_steps)
    loss_fct = CrossEntropyLoss(ignore_index=0)

    def calc_batch_loss(batch):
        # データ型をGPU/CPUにあわせて変更
        batch = tuple(t.to(device) for t in batch)
        input_ids, y, input_mask, input_type_id, masked_pos = batch

        # モデルから出力を計算
        logits = model(input_ids, input_type_id, input_mask)
        logits = logits.view(-1, tokenizer.vocab_size)
        y = y.view(-1)

        # 損失を計算
        loss = loss_fct(logits, y)
        return loss

    logger.info("train starts")
    # モデルを学習モードに変更
    model.train()
    # ログの出力先を指定
    summary_writer = SummaryWriter(log_dir="logs")

    generated_texts = []
    try:
        # trangeは進捗バー表示ができる便利ツール
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            train_loss = 0.
            running_num = 0

            # 教師データをバッチ別に処理していく
            for step, batch in enumerate(train_dataloader):

                # 損失を計算
                loss = calc_batch_loss(batch)

                # 勾配を計算、パラメーターを更新
                loss.backward()
                optimizer.step()
                scheduler.step()

                # 勾配は適宜初期化？
                optimizer.zero_grad()

                # 出力用、損失の合計とステップ数
                train_loss += loss.item()
                running_num += len(batch[0])
            logger.info("[{0} epochs] "
                        "train loss: {1:.3g} ".format(epoch + 1,
                                                      train_loss / running_num))
            summary_writer.add_scalar("train_loss",
                                      train_loss / running_num, epoch)

            # モデルを評価モードにする
            model.eval()
            valid_loss = 0.
            valid_num = 0
            for batch in valid_dataloader:
                # 評価データに対して予測を適用、出力用
                valid_loss += calc_batch_loss(batch).item()
                valid_num += len(batch[0])

            # 出力用リストにデータを追加
            generated_texts.append(generate(tokenizer=tokenizer,
                                            device=device,
                                            length=25,
                                            max_length=args.max_seq_length,
                                            model=model))
            logger.info("[{0} epochs] valid loss: {1:.3g}".format(epoch + 1,
                                                                  valid_loss / valid_num))
            summary_writer.add_scalar("val_loss",
                                      valid_loss / valid_num, epoch)

            # 学習モードに再度切り替え
            model.train()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt")

    # ログ出力を終了
    summary_writer.close()
    dt_now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    # モデルを保存
    save(args, model, tokenizer, str(dt_now))


def initialization_text(tokenizer, length, fix_word):
    # 除外するヘッダ等トークン
    except_tokens = ["[MASK]", "[PAD]", "[UNK]", "[CLS]", "[SEP]"]
    except_ids = [tokenizer.vocab[token] for token in except_tokens]

    # vocab_sizeのうち、except_ids以外は、利用する
    candidate_ids = [i for i in range(tokenizer.vocab_size)
                     if i not in except_ids]

    init_tokens = []
    # ヘッダ
    init_tokens.append(tokenizer.vocab["[CLS]"])
    for _ in range(length):
        # ランダムに文字を選択
        init_tokens.append(random.choice(candidate_ids))
    # フッタ
    init_tokens.append(tokenizer.vocab["[SEP]"])

    return init_tokens


def generate(tokenizer, device, max_iter=10, length=50, max_length=128,
             model=None, fix_word=None, samples=1):
    if isinstance(model, str):
        # モデルがstrで指定されていれば

        # モデルを読み込む、state_dictはパラメータを読み込む？
        model_state_dict = torch.load(os.path.join(model, "pytorch_model.bin"),
                                      map_location=device)
        model = BertMouth.from_pretrained(model,
                                          state_dict=model_state_dict,
                                          num_labels=tokenizer.vocab_size)
        # GPU/CPUにあわせる
        model.to(device)

    for _ in range(samples):
        # 指定したサンプル回数生成する

        # ランダムな単語列を生成
        generated_token_ids = initialization_text(tokenizer, length, fix_word)

        if fix_word:
            # もし事前に文字列が指定されていれば
            # 文字列を分かち書きする
            tokenized_fix_word = tokenizer.tokenize(fix_word)
            # 行頭から行末までのランダムな位置を決めて、fix_wordをそこに配置する
            fix_word_pos = random.randint(1,
                                          length - len(tokenized_fix_word))
            # fix_word_intervalを、後の処理でfix_wordを削除しないように保持しておく
            fix_word_interval = set(range(fix_word_pos,
                                          fix_word_pos + len(tokenized_fix_word)))
            for i in range(len(tokenized_fix_word)):
                generated_token_ids[fix_word_pos + i] = \
                    tokenizer.convert_tokens_to_ids(tokenized_fix_word[i])

        else:
            fix_word_interval = []

        # input_type_idをmax_length分で0
        # generated_token_idsをmax_length分で0埋め
        # input_maskも同様に[1, 1, 1, 1... 0, 0, 0]
        # テンソルに型変換
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

        # 
        pre_tokens = generated_token_ids.clone()
        for _ in range(max_iter):
            for j in range(length):
                if fix_word_interval:
                    if j + 1 in fix_word_interval:
                        # fix_wordは文字を入れ替えないので処理をパス
                        continue

                # 文章のトークン１つをMASKに変換する
                generated_token_ids[0, j + 1] = tokenizer.vocab["[MASK]"]

                # 予測、(all_encoder_layers, _)が返り値
                logits = model(generated_token_ids,
                               input_type_id, input_mask)[0]

                # MASKにした箇所で最も確率が高いトークンを取得し、置き換える
                sampled_token_id = torch.argmax(logits[j + 1])
                generated_token_ids[0, j + 1] = sampled_token_id

            # idから文字列に変換、結合
            sampled_sequence = [tokenizer.ids_to_tokens[token_id]
                                for token_id in generated_token_ids[0].cpu().numpy()]
            sampled_sequence = "".join([token[2:] if token.startswith("##") else token
                                        for token in sampled_sequence[1:length + 1]])
            if torch.equal(pre_tokens, generated_token_ids):
                # 文字列が同じになったら生成を中止
                break
            pre_tokens = generated_token_ids.clone()
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

    if args.do_train:
        # 学習
        train(args, tokenizer, device)
    if args.do_generate:
        # 生成
        generate(tokenizer, device, max_iter=args.max_iter,
                 length=args.seq_length, model=args.bert_model,
                 fix_word=args.fix_word, samples=args.samples)


if __name__ == '__main__':
    main()
