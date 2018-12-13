#!/usr/bin/env python2.7

import os
import random
import numpy as np

from PIL import Image

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import nltk
import random
import tflearn
from flask import Flask
from nltk.corpus import stopwords
from tensorflow.keras import models
from tensorflow.keras import layers
#from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from nltk.stem.snowball import SnowballStemmer

from grpc.beta import implementations

from mnist import MNIST # pylint: disable=no-name-in-module

TF_MODEL_SERVER_HOST = os.getenv("TF_MODEL_SERVER_HOST", "127.0.0.1")
TF_MODEL_SERVER_PORT = int(os.getenv("TF_MODEL_SERVER_PORT", 9000))
TF_DATA_DIR = os.getenv("TF_DATA_DIR", "/tmp/data/")
TF_MNIST_IMAGE_PATH = os.getenv("TF_MNIST_IMAGE_PATH", None)
TF_MNIST_TEST_IMAGE_NUMBER = int(os.getenv("TF_MNIST_TEST_IMAGE_NUMBER", -1))
#BOWORDS_PATH = os.getenv("BOWORDS_PATH", "/mnt/testdata")
BOWORDS_PATH = os.getenv("BOWORDS_PATH", ".")
BOWORDS_FILE = os.getenv("BOWORDS__FILE", "global_words.txt")

stop_words = stopwords.words('english')
nltk.download('stopwords')
nltk.download('punkt')
stemmer = SnowballStemmer("english", ignore_stopwords=True)
tf.logging.set_verbosity(tf.logging.ERROR)

words = sorted(list(set(open(BOWORDS_PATH +'/'+BOWORDS_FILE, 'r'))))

def get_tf_record(sentence):

    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words if not word in stop_words]
    bow = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w.strip() == s.strip():
                bow[i] = 1

    return(np.array(bow))

query = "Find all sites"

x = get_tf_record(query)

print(x)

channel = implementations.insecure_channel(
    TF_MODEL_SERVER_HOST, TF_MODEL_SERVER_PORT)
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

request = predict_pb2.PredictRequest()
request.model_spec.name = "helix"
request.model_spec.signature_name = "predict_class"
request.inputs['query'].CopyFrom(
    tf.contrib.util.make_tensor_proto(x, shape=[1, len(words)]))
#request.inputs['classes'].CopyFrom(
 #   tf.contrib.util.make_tensor_proto(np.array(1,2,3), shape=[3]))

result = stub.Predict(request, 10.0)  # 10 secs timeout

print(result)
print(MNIST.display(image, threshold=0))
print("Your model says the above number is... %d!" %
      result.outputs["classes"].int_val[0])
