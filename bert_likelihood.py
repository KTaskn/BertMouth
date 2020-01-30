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

    def yomi(self, text):
        return list(map(lambda x: x.yomi, self.juman.analysis(text).mrph_list()))

    def _remove_small(self, token):
        l_small = ['ぁ', 'ぃ', 'ぅ', 'ぇ', 'ぉ', 'ゃ', 'ゅ', 'ょ', 'っ']
        for small in l_small:
            token = token.replace(small, '')
        return token


    def remove_small(self, tokens):
        return list(map(self._remove_small, tokens))

    def tanka_score_subsets(self, text):
        _count = [5, 7]
        score = 0
        REWARD = 1
        PENALTY = -1

        for row in self.subset(self.remove_small(self.yomi(text))):
            if len(''.join(row)) in _count:
                score += REWARD
            else:
                score += PENALTY
        return score

    def tanka_score_flow(self, text):
        _count = [5, 12, 17, 24, 31]
        REWARD = [5, 10, 15, 20, 25, 1000]

        score = 0
        idx = 0

        subset = ''
        for row in self.remove_small(self.yomi(text)):
            subset += row
            if len(subset) == _count[idx]:
                score = REWARD[idx]
                idx += 1
            elif len(subset) < _count[idx]:
                pass
            else:
                break
            
        return score

    def subset(self, tokens):
        subsets = []

        length = len(tokens)
        for idx_a in range(length):
            for idx_b in range(idx_a + 1, length + 1):
                subsets.append(tokens[idx_a:idx_b])

        return subsets



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
        "[PAD]",
        "[UNK]", "[CLS]", "[SEP]",
        "（", "）", "・", "／", "、", "。", "！", "？", "「", "」", "…", "’", "』", "『", "：", "※"
        ]
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

    def tokens2text(self, tokens):
        sampled_sequence = [self.bert_tokenizer.ids_to_tokens[token_id]
                                        for token_id in tokens[0].cpu().numpy()]
        sampled_sequence = "".join(
            [
                token[2:] if token.startswith("##") else token
                for token in list(filter(lambda x: x != '[PAD]' and x != '[CLS]' and x != '[SEP]', sampled_sequence))
            ]
        )
        return sampled_sequence


    def likelihood(self, tokens):
        outputs = self.model(tokens)
        predictions = outputs[0]

        score_sum = 0.0
        for idx, scores in zip(tokens[0].tolist(), predictions[0].tolist()):
            score_sum += scores[idx]
        return score_sum

    def initialization_text(self, length=10):
        init_tokens = []
        # ヘッダ
        init_tokens.append(self.bert_tokenizer.vocab["[CLS]"])
        for _ in range(length):
            # ランダムに文字を選択
            init_tokens.append(random.choice(self.candidate_ids))
        # フッタ
        init_tokens.append(self.bert_tokenizer.vocab["[SEP]"])

        return torch.tensor(init_tokens).reshape(1, -1)

    def scoring(self, tokens):
        return self.likelihood(tokens) + self.juman_tokenizer.tanka_score_subsets(self.tokens2text(tokens)) + self.juman_tokenizer.tanka_score_flow(self.tokens2text(tokens))

    def select(self, l_tokens, size=5):
        scores = list(map(self.scoring, l_tokens))
        print(sorted(scores, reverse=True)[:3])
        selected = list(map(
            lambda x: x[0],
            sorted(
                list(zip(l_tokens, scores)), 
                key=lambda x: x[1],
                reverse=True
            )
        ))

        return selected

    def crossover(self, tokens_0, tokens_1):
        l_tokens_0 = tokens_0.numpy().reshape(-1).tolist()
        l_tokens_1 = tokens_1.numpy().reshape(-1).tolist()

        start = random.randint(1, len(l_tokens_0) - 3)
        end = random.randint(start, len(l_tokens_0) - 2)

        for num in range(start, end):
            l_tokens_0[num] = l_tokens_1[num]

        return torch.tensor(l_tokens_0).reshape(1, -1)

    def mutation(self, tokens, N=3):
        l_tokens = tokens.numpy().reshape(-1).tolist()

        for num in range(N):
            num = random.randint(1, len(l_tokens) - 2)
            l_tokens[num] = self.bert_tokenizer.vocab["[MASK]"]
            
            outputs = self.model(torch.tensor(l_tokens).reshape(1, -1))
            predictions = outputs[0]
            _, predicted_indexes = torch.topk(predictions[0, num], k=9)

            random_tokens = [random.choice(self.candidate_ids) for i in range(1)]

            predicted_indexes = list(
                set(predicted_indexes.tolist() + random_tokens) - set(self.except_ids)
            )

            predicted_tokens = self.bert_tokenizer.convert_ids_to_tokens(predicted_indexes)
            predict_token = random.choice(predicted_indexes)

            l_tokens[num] = predict_token

        return torch.tensor(l_tokens).reshape(1, -1)



if __name__ == '__main__':
    import pandas as pd
    from pprint import pprint
    args = parse_argument()
    gen = Generater(args.bert_path)

    epoch = 100
    N = 100
    S = 20

    TOP = 0

    l_tokens = [gen.initialization_text() for i in range(N)]

    for idx in range(epoch):
        selected = gen.select(l_tokens, size=S)

        pprint(list(map(
            gen.tokens2text,
            selected[:5]
        )))

        l_tokens = selected[:TOP]

        for n in range(N - TOP):
            l_tokens.append(
                gen.mutation(gen.crossover(
                    random.choice(selected),
                    random.choice(selected)
                ), random.choice([1, 1, 1, 2, 3]))
            )

        

