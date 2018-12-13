import os
import random
import numpy
import logging
import sys

import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import nltk
import tflearn
import numpy as np
from flask import Flask
from nltk.corpus import stopwords
from tensorflow.keras import models
from tensorflow.keras import layers
#from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from nltk.stem.snowball import SnowballStemmer

from grpc.beta import implementations
from flask import Flask, render_template, request, jsonify

app = Flask(__name__,static_url_path='/static', static_folder='static')
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.DEBUG)

TF_MODEL_SERVER_HOST = os.getenv("TF_MODEL_SERVER_HOST", "127.0.0.1")
TF_MODEL_SERVER_PORT = int(os.getenv("TF_MODEL_SERVER_PORT", 9000))

BOWORDS_PATH = os.getenv("BOWORDS_PATH", "/app/global_words.txt")

stop_words = stopwords.words('english')
stemmer = SnowballStemmer("english", ignore_stopwords=True)
tf.logging.set_verbosity(tf.logging.DEBUG)

words = sorted(list(set(open(BOWORDS_PATH, 'r'))))

def get_tf_record(sentence):

    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words if not word in stop_words]
    bow = [0]*len(words)
    for s in sentence_words:
        print(s)
        for i, w in enumerate(words):
            if w.strip() == s.strip():
                bow[i] = 1

    return(np.array(bow))


@app.route("/")
def main():
    return render_template('main.html')

@app.route('/classify', methods=['POST'])
def predict():

    predicted_class = -1
    from flask import request
    if request.method == "POST":
        # get url
        print(request.form)
        query = request.get_json(force=True)
        print(query)
        x = get_tf_record(str(query.get("query")))

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
        arr = tf.contrib.util.make_ndarray(result.outputs["classes"])[0]
        print(arr)
        predicted_class = np.argmax(arr)
        print(predicted_class)
    return str(predicted_class)

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=5000)
