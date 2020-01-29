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
    parser.add_argument("--target", default=None, type=str, required=True,
                        help="A fixed word in text generation.")
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


class BertWithJumanModel():
    def __init__(self, bert_path, vocab_file_name="vocab.txt", use_cuda=False):
        # 日本語文章をBERTに食わせるためにJumanを読み込む
        self.juman_tokenizer = JumanTokenizer()
        # 事前学習済みのBERTモデルを読み込む
        self.model = BertModel.from_pretrained(bert_path)
        # 事前学習済みのBERTモデルのTokenizerを読み込む
        self.bert_tokenizer = BertTokenizer(Path(bert_path) / vocab_file_name,
                                            do_lower_case=False, do_basic_tokenize=False)
        # CUDA-GPUを利用するかどうかのフラグ読み込み
        self.use_cuda = use_cuda

    def _preprocess_text(self, text):
        # 事前処理、テキストの半角スペースは削除
        try:
            return text.replace(" ", "")  # for Juman
        except:
            return ''

    def get_sentence_embedding(self, text, pooling_layer=-2, pooling_strategy="REDUCE_MEAN"):
        # テキストの半角スペースを削除する
        preprocessed_text = self._preprocess_text(text)
        # 日本語のテキストを分かち書きし、トークンリストに変換する
        tokens = self.juman_tokenizer.tokenize(preprocessed_text)
        # トークンを半角スペースで結合しstrに変換する
        bert_tokens = self.bert_tokenizer.tokenize(" ".join(tokens))
        # テキストのサイズは128までなので、ヘッダ + トークン126個 + フッタを作成
        # トークンをidに置換する
        ids = self.bert_tokenizer.convert_tokens_to_ids(["[CLS]"] + bert_tokens[:126] + ["[SEP]"]) # max_seq_len-2
        tokens_tensor = torch.tensor(ids).reshape(1, -1)

        return self.score(tokens_tensor, pooling_layer=pooling_layer, pooling_strategy=pooling_strategy)

    def score(self, tokens_tensor, pooling_layer=-2, pooling_strategy="REDUCE_MEAN"):
        if self.use_cuda:
            # GPUの利用チェック、利用
            tokens_tensor = tokens_tensor.to('cuda')
            self.model.to('cuda')

        # モデルを評価モードに変更
        self.model.eval()
        with torch.no_grad():
            # 自動微分を適用しない（メモリ・高速化などなど）
            # id列からベクトル表現を計算する
            all_encoder_layers, _ = self.model(tokens_tensor)

            # SWEMと同じ方法でベクトルを時間方向にaverage-poolingしているらしい
            # 文章列によって次元が可変になってしまうので、伸びていく方向に対してプーリングを行い次元を固定化する
            # https://yag-ays.github.io/project/swem/
            embedding = all_encoder_layers[pooling_layer].cpu().numpy()[0]
            if pooling_strategy == "REDUCE_MEAN":
                return np.mean(embedding, axis=0)
            elif pooling_strategy == "REDUCE_MAX":
                return np.max(embedding, axis=0)
            elif pooling_strategy == "REDUCE_MEAN_MAX":
                return np.r_[np.max(embedding, axis=0), np.mean(embedding, axis=0)]
            elif pooling_strategy == "CLS_TOKEN":
                return embedding[0]
            else:
                raise ValueError("specify valid pooling_strategy: {REDUCE_MEAN, REDUCE_MAX, REDUCE_MEAN_MAX, CLS_TOKEN}")



class Generater:
    def __init__(self, tokenizer, bert_path):
        self.tokenizer = tokenizer
        self.vocab_size = len(self.tokenizer.vocab)

        # 事前学習済みのBERTモデルのMaskedLMタスクモデルを読み込む
        self.model = BertForMaskedLM.from_pretrained(bert_path)

        # 除外するヘッダ等トークン
        except_tokens = ["[MASK]", 
        #"[PAD]",
        "[UNK]", "[CLS]", "[SEP]"]
        self.except_ids = [self.tokenizer.vocab[token] for token in except_tokens]

        # vocab_sizeのうち、except_ids以外は、利用する
        self.candidate_ids = [i for i in range(self.vocab_size)
                        if i not in self.except_ids]

    def initialization_text(self, length):
        init_tokens = []
        # ヘッダ
        init_tokens.append(self.tokenizer.vocab["[CLS]"])
        for _ in range(length):
            # ランダムに文字を選択
            init_tokens.append(random.choice(self.candidate_ids))
        # フッタ
        init_tokens.append(self.tokenizer.vocab["[SEP]"])

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
        l_tokens[num] = self.tokenizer.vocab["[MASK]"]
        
        outputs = self.model(torch.tensor(l_tokens).reshape(1, -1))
        predictions = outputs[0]
        _, predicted_indexes = torch.topk(predictions[0, num], k=10)

        predicted_indexes = list(set(predicted_indexes.tolist()) - set(self.except_ids))

        predicted_tokens = self.tokenizer.convert_ids_to_tokens(predicted_indexes)
        predict_token = random.choice(predicted_indexes)

        l_tokens[num] = predict_token

        return torch.tensor(l_tokens).reshape(1, -1)



if __name__ == '__main__':
    import pandas as pd
    args = parse_argument()
    bwjm = BertWithJumanModel(
        bert_path=args.bert_path
    )
    vec_target = bwjm.get_sentence_embedding(args.target)

    l_tokens = []
    vecs_gen = []
    epoch = 1000

    N = 100
    NC = 30

    LENGTH = 20

    gen = Generater(bwjm.bert_tokenizer, args.bert_path)
    for num in range(N):
        tokens = gen.initialization_text(LENGTH)
        l_tokens.append(tokens)

    for i in range(epoch):
        vecs_gen = []
        for tokens in l_tokens:
            vecs_gen.append(bwjm.score(tokens))

        selected = gen.select(vec_target, vecs_gen, size=NC)
        l_tokens_tmp = []

        for idx in selected:
            l_tokens_tmp.append(l_tokens[idx])

        l_tokens = l_tokens_tmp

        for num in range(N - NC):
            l_tokens.append(
                gen.mutation(gen.crossover(
                    random.choice(l_tokens_tmp),
                    random.choice(l_tokens_tmp)
                ))
            )

        tokens = l_tokens[0]
        sampled_sequence = [bwjm.bert_tokenizer.ids_to_tokens[token_id]
                                            for token_id in tokens[0].cpu().numpy()]
        sampled_sequence = "".join(
            [
                token[2:] if token.startswith("##") else token
                for token in list(filter(lambda x: x != '[PAD]', sampled_sequence))
            ]
        )
        print(sampled_sequence)