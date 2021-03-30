import argparse
import json
import os
import sys
import unicodedata

from flask import Flask, request

import numpy as np
import tensorflow as tf

from data import Word, Sentence, Document, Token, PREDICTION_SUMMARIZERS
from example import EXAMPLE_GENERATORS, examples_to_inputs
from label import LabelEncoder, IobesTokenLabeler, iobes_to_iob2, LABEL_ASSIGNERS
from model import load_ner_model


app = Flask(__name__)


@app.route('/', methods=["POST"])
def tag():
    data = request.get_json()
    text = data['text']
    return app.tagger.tag(text)


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

    def tag(self, text):
        words = text.split()
        docs = [Document([Sentence([Word(token, "O") for token in words])])]

        docs[0].tokenize(self.tokenizer.tokenize, self.token_labeler.label_tokens)
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

        return_words, return_labels = [], []
        for sentence in docs[0].sentences:
            for word in sentence.words:
                return_labels.append(word.predicted_label)
                return_words.append(word.text)

        return json.dumps(
            {"tokens": return_words, "labels": return_labels},
            ensure_ascii=False
        )

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
