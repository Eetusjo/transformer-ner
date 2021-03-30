import argparse
import json
import os
import sys
import unicodedata
import numpy as np
import tensorflow as tf

from flask import Flask, request
from data import Word, Sentence, Document, Token, PREDICTION_SUMMARIZERS
from example import EXAMPLE_GENERATORS, examples_to_inputs
from label import LabelEncoder, IobesTokenLabeler, iobes_to_iob2, LABEL_ASSIGNERS
from model import load_ner_model
from mosestokenizer import MosesTokenizer, MosesSentenceSplitter


app = Flask(__name__)


@app.route('/', methods=["POST"])
def tag():
    data = request.get_json()
    text = data["text"]
    mode = data["mode"] if "mode" in data else "sentence"
    output = data["output"] if "output" in data else "doc"
    return app.tagger.tag(text, mode, output)


class Tagger(object):
    def __init__(self, model, decoder, tokenizer, word_labels, config, saved_args):
        self.model = model
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.word_labels = word_labels
        self.config = config
        self.saved_args = saved_args

        self.max_len = saved_args["max_seq_length"]

        self.token_labeler = IobesTokenLabeler(word_labels)
        self.label_encoder = LabelEncoder(self.token_labeler.labels())

        self.encode_tokens = lambda t: self.tokenizer.encode(t, add_special_tokens=False)

        self.assign_labels = LABEL_ASSIGNERS[saved_args["assign_labels"]]
        self.example_generator = EXAMPLE_GENERATORS[saved_args["examples"]](
            self.max_len,
            Token(self.tokenizer.cls_token, is_special=True, masked=False),
            Token(self.tokenizer.sep_token, is_special=True, masked=False),
            Token(self.tokenizer.pad_token, is_special=True, masked=True),
            self.encode_tokens,
            self.label_encoder.encode
        )
        self.summarize_predictions = PREDICTION_SUMMARIZERS[saved_args["summarize_preds"]]

        self.splitter = MosesSentenceSplitter("fi")
        self.pretokenizer = MosesTokenizer("fi")

    def tag(self, text, mode, output):
        if mode == "sentence":
            words = self.pretokenizer(text)
            docs = [Document([Sentence([Word(token, "O") for token in words])])]
        elif mode == "doc":
            sent_objs = []
            sents = self.splitter([text])
            for sent in sents:
                sent_objs.append(Sentence([Word(token, "O") for token in self.pretokenizer(sent)]))
            docs = [Document(sent_objs)]
        else:
            return "{}"

        for d in docs:
            d.tokenize(self.tokenizer.tokenize, self.token_labeler.label_tokens)

        examples = self.example_generator.examples(docs)
        examples = list(examples)
        x, y = examples_to_inputs(examples)
        test_predictions = self.model.predict(x)

        for example, preds in zip(examples, test_predictions):
            assert len(example.tokens) == len(preds)
            for pos, (token, pred) in enumerate(zip(example.tokens, preds)):
                token.predictions.append((pos, pred))

        for document in docs:
            self.summarize_predictions(document)
            self.assign_labels(document, self.label_encoder)

        if output == "json":
            return_words, return_labels = [], []
            for sentence in docs[0].sentences:
                return_words.append([]), return_labels.append([])
                for word in sentence.words:
                    return_labels[-1].append(word.predicted_label)
                    return_words[-1].append(word.text)

            return json.dumps(
                {"tokens": return_words, "labels": return_labels},
                ensure_ascii=False
            ) + "\n"
        elif output == "tsv":
            outs = ""
            for i, sentence in enumerate(docs[0].sentences):
                for word in sentence.words:
                    outs += f"{word.text}\t{word.predicted_label}\n"
                outs += "\n"
            return outs
        else:
            return "{}"


    @classmethod
    def load(cls, model_dir):
        model, decoder, tokenizer, word_labels, config, saved_args = load_ner_model(model_dir)
        tagger = cls(model, decoder, tokenizer, word_labels, config, saved_args)
        return tagger



def main(argv):
    parser = argparse.ArgumentParser('serve')
    parser.add_argument("ner_model_dir")
    args = parser.parse_args()

    app.tagger = Tagger.load(args.ner_model_dir)
    app.run(host='0.0.0.0', port=8080)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
