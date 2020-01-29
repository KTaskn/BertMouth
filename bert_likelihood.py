from pathlib import Path
import argparse
from tqdm import tqdm, trange
import random

import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
from pytorch_transformers import BertForMaskedLM
from pyknp import Juman

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default=None, type=str, required=True,
                        help="text")
    parser.add_argument("--bert_path", default=None, type=str,
                        help="bert path")

    args = parser.parse_args()
    return args


class JumanTokenizer():
    def __init__(self):
        self.juman = Juman()

    def tokenize(self, text):
        # Jumanを用いて、日本語の文章を分かち書きする。
        result = self.juman.analysis(text)
        return [mrph.midasi for mrph in result.mrph_list()]


class Generater:
    def __init__(self, bert_path):
        vocab_file_name = 'vocab.txt'
        # 日本語文章をBERTに食わせるためにJumanを読み込む
        self.juman_tokenizer = JumanTokenizer()
        # 事前学習済みのBERTモデルを読み込む
        self.model = BertModel.from_pretrained(bert_path)
        # 事前学習済みのBERTモデルのTokenizerを読み込む
        self.bert_tokenizer = BertTokenizer(Path(bert_path) / vocab_file_name,
                                            do_lower_case=False, do_basic_tokenize=False)
        self.vocab_size = len(self.bert_tokenizer.vocab)

        # 事前学習済みのBERTモデルのMaskedLMタスクモデルを読み込む
        self.model = BertForMaskedLM.from_pretrained(bert_path)

        # 除外するヘッダ等トークン
        except_tokens = ["[MASK]", 
        #"[PAD]",
        "[UNK]", "[CLS]", "[SEP]"]
        self.except_ids = [self.bert_tokenizer.vocab[token] for token in except_tokens]

        # vocab_sizeのうち、except_ids以外は、利用する
        self.candidate_ids = [i for i in range(self.vocab_size)
                        if i not in self.except_ids]


    def _preprocess_text(self, text):
        # 事前処理、テキストの半角スペースは削除
        return text.replace(" ", "").replace('#', '')  # for Juman

    def text2tokens(self, text):
        # テキストの半角スペースを削除する
        preprocessed_text = self._preprocess_text(text)
        # 日本語のテキストを分かち書きし、トークンリストに変換する
        tokens = self.juman_tokenizer.tokenize(preprocessed_text)
        # トークンを半角スペースで結合しstrに変換する
        bert_tokens = self.bert_tokenizer.tokenize(" ".join(tokens))
        # テキストのサイズは128までなので、ヘッダ + トークン126個 + フッタを作成
        # トークンをidに置換する
        ids = self.bert_tokenizer.convert_tokens_to_ids(["[CLS]"] + bert_tokens[:126] + ["[SEP]"]) # max_seq_len-2
        generated_token_ids = torch.tensor(ids).reshape(1, -1)
        return generated_token_ids

    def likelihood(self, tokens):
        outputs = self.model(tokens)
        predictions = outputs[0]

        score_sum = 0.0
        for idx, scores in zip(tokens[0].tolist(), predictions[0].tolist()):
            score_sum += scores[idx]
        return score_sum

    def initialization_text(self, length):
        init_tokens = []
        # ヘッダ
        init_tokens.append(self.bert_tokenizer.vocab["[CLS]"])
        for _ in range(length):
            # ランダムに文字を選択
            init_tokens.append(random.choice(self.candidate_ids))
        # フッタ
        init_tokens.append(self.bert_tokenizer.vocab["[SEP]"])

        return torch.tensor(init_tokens).reshape(1, -1)

    def select(self, vec_target, vecs_gen, size=5):
        ret = []
        for vec in vecs_gen:
            ret.append(np.linalg.norm(vec_target - vec))
        print(ret)
        return list(map(
            lambda x: x[1],
            sorted(tuple(zip(ret, range(len(vecs_gen)))), key=lambda x: x[0])[:size]
        ))

    def crossover(self, tokens_0, tokens_1):
        l_tokens_0 = tokens_0.numpy().reshape(-1).tolist()
        l_tokens_1 = tokens_1.numpy().reshape(-1).tolist()

        start = random.randint(1, len(l_tokens_0) - 3)
        end = random.randint(start, len(l_tokens_0) - 2)

        for num in range(start, end):
            l_tokens_0[num] = l_tokens_1[num]

        return torch.tensor(l_tokens_0).reshape(1, -1)

    def mutation(self, tokens):
        l_tokens = tokens.numpy().reshape(-1).tolist()
        num = random.randint(1, len(l_tokens) - 2)
        l_tokens[num] = self.bert_tokenizer.vocab["[MASK]"]
        
        outputs = self.model(torch.tensor(l_tokens).reshape(1, -1))
        predictions = outputs[0]
        _, predicted_indexes = torch.topk(predictions[0, num], k=10)

        predicted_indexes = list(set(predicted_indexes.tolist()) - set(self.except_ids))

        predicted_tokens = self.bert_tokenizer.convert_ids_to_tokens(predicted_indexes)
        predict_token = random.choice(predicted_indexes)

        l_tokens[num] = predict_token

        return torch.tensor(l_tokens).reshape(1, -1)



if __name__ == '__main__':
    import pandas as pd
    args = parse_argument()
    gen = Generater(args.bert_path)

    tokens = gen.text2tokens(args.text)
    print(gen.likelihood(tokens))