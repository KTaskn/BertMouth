
import logging
from pathlib import Path
import argparse

import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
from pytorch_transformers import BertForMaskedLM
from pyknp import Juman


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
    parser.add_argument("--no_cuda",
                    action='store_true',
                    help="Whether not to use CUDA when available.")

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
        # 事前学習済みのBERTモデルのMaskedLMタスクモデルを読み込む
        self.model = BertForMaskedLM.from_pretrained(bert_path)
        # 事前学習済みのBERTモデルのTokenizerを読み込む
        self.bert_tokenizer = BertTokenizer(Path(bert_path) / vocab_file_name,
                                            do_lower_case=False, do_basic_tokenize=False)
        # CUDA-GPUを利用するかどうかのフラグ読み込み
        self.use_cuda = use_cuda

    def _preprocess_text(self, text):
        # 事前処理、テキストの半角スペースは削除
        return text.replace(" ", "")  # for Juman

    def paraphrase(self, text):
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

        if self.use_cuda:
            # GPUの利用チェック、利用
            generated_token_ids = generated_token_ids.to('cuda')
            self.model.to('cuda')

        # モデルを評価モードに変更
        self.model.eval()
        with torch.no_grad():
            for i in range(10):
                for j, _ in enumerate(tokens):
                    # 文章のトークン１つをMASKに変換する
                    # ヘッダは除くから、+1から
                    masked_index = j + 1

                    pre_token = generated_token_ids[0, masked_index].item()

                    generated_token_ids[0, masked_index] = self.bert_tokenizer.vocab["[MASK]"]

                    outputs = self.model(generated_token_ids)
                    predictions = outputs[0]

                    _, predicted_indexes = torch.topk(predictions[0, masked_index], k=5)
                    predicted_tokens = self.bert_tokenizer.convert_ids_to_tokens(predicted_indexes.tolist())

                    print(predicted_tokens)

                    predict_token = predicted_indexes.tolist()[0]

                    # if pre_token == predict_token:
                    #     predict_token = predicted_indexes.tolist()[1]

                    generated_token_ids[0, masked_index] = predict_token

                    # idから文字列に変換、結合
                    sampled_sequence = [self.bert_tokenizer.ids_to_tokens[token_id]
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

    bwjm = BertWithJumanModel(
        bert_path=args.bert_model,
        use_cuda=args.no_cuda == False
    )
    bwjm.paraphrase(args.fix_word)



if __name__ == '__main__':
    main()
