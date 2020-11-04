#!/usr/bin/env python


from flask import Flask, request
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer


server = Flask(__name__)


@server.route("/ping")
def ping():
    return '', 200


@server.route("/invocations")
def invocations():
    task = request.args.get('task', None)
    if task == 'nli':
        str_a = request.args.get('str_a', None)
        str_b = request.args.get('str_b', None)
        output = get_nli_result(str_a, str_b)

    elif task == 'embed':
        sentence = request.args.get('sentence', None)
        output = get_embedding(sentence)

    return output


def get_embedding(sentence):
    embedding = sentence_embedder.encode([sentence])
    return str(embedding)


def get_nli_result(str_a, str_b):
    inputs = nli_tokenizer(str_a + str_b, return_tensors="pt").to(device)
    outputs = nli_model(**inputs)
    probs = nn.Softmax(dim=1)(outputs[0])[0]
    probs = probs.cpu()

    return str(probs)


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    nli_tokenizer = AutoTokenizer.from_pretrained("models/roberta-mnli-tokenizer")
    nli_model = AutoModelForSequenceClassification.from_pretrained(
        "models/roberta-mnli-model")
    nli_model.to(device)

    sentence_embedder = SentenceTransformer("models/distilbert")

    server.run(host='0.0.0.0', port=8080)
