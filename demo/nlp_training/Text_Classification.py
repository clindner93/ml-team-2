import tensorflow as tf
from tensorflow import keras
import json
import numpy as np
import nltk
import random
import tflearn
from flask import Flask
from nltk.corpus import stopwords
from tensorflow.keras import models
from tensorflow.keras import layers
#from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from nltk.stem.snowball import SnowballStemmer
import os
import sys
#from keras import backend as K
#from scipy import metrics
#from sklearn import metrics

print("helix version 4")

TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_FILE_PATH", "./TrainingDataset.json")

#PATH FOR TEST DATA
TEST_DATA_PATH = os.getenv("TEST_DATA_PATH", "./TestDataset.txt")

#PATH FOR TEST DATA
TEST_CLASSES_PATH = os.getenv("TEST_CLASSES_PATH", "./TestDataClasses.txt")

#PATH FOR MODEL
TF_MODEL_VERSION = os.getenv("TF_MODEL_VERSION","1")
TF_MODEL_EXPORT_PATH = os.getenv("TF_MODEL_EXPORT_PATH", "./modelpath")

#PATH FOR BAG OF WORDS
BOWORDS_PATH = os.getenv("BOWORDS_PATH", "./global_words.txt")

MODEL_PATH = TF_MODEL_EXPORT_PATH + "/" + TF_MODEL_VERSION


def get_tf_record(sentence):
    global words
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words if not word in stop_words]
    bow = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w.strip() == s.strip():
                bow[i] = 1
    return(np.array(bow))


def classifyQuery(model, query=None):

    prediction = model.predict([get_tf_record(query)])[0]
    if prediction[np.argmax(model.predict([get_tf_record(query)]))] >= 0.85:
        return (classes[np.argmax(prediction)])
        
    return none



if __name__ == '__main__':
    success = 0
    query_file = open(TEST_DATA_PATH, 'r') 
    class_file = open(TEST_CLASSES_PATH,'r')
    indx = 0
    classified_classes = ['Unknown'] * 16

    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    tf.logging.set_verbosity(tf.logging.DEBUG)

    with open(TRAIN_DATA_PATH) as json_data:
        data = json.load(json_data)

    classes = data.keys()
    print("The classes are " + str(classes))
    print("The number of classes are " + str(len(classes)))



    for length in range(0, len(classes)):
        print("Class " + str(length+1) + " is " + str(classes[length]) + " - It has " + str(len(data[classes[length]])) + " TrainingDataset")


    words = set([])
    docs = []


    for each_class in data.keys():
        for each_sentence in data[each_class]:
            w = nltk.word_tokenize(each_sentence)
            for wo in w:
                words.add(wo)
            docs.append((w, each_class))


    stop_words = stopwords.words('english')

    words = [stemmer.stem(w.lower()) for w in words]
    words = [w for w in words if not w in stop_words] 
    words = sorted(list(set(words)))

    training = []
    output = []
    output_empty = [0] * len(classes)

    for doc in docs:

        bow = []
        token_words = doc[0]
        token_words = [stemmer.stem(word.lower()) for word in token_words if not word in stop_words]
        for w in words:
            bow.append(1) if w in token_words else bow.append(0)
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bow, output_row])


    with open(BOWORDS_PATH, 'w') as f:
        for word in words:
            f.write("%s\n" % word)

    random.shuffle(training)
    training = np.array(training)

    train_x = list(training[:, 0])
    print(len(train_x[0]))

    train_y = list(training[:, 1])

    ## Define model
    tf.reset_default_graph() 
    norm_init_with_seed = tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=1,dtype=tf.float32)
    net = tflearn.input_data(shape=[None, len(train_x[0])])
    net = tflearn.fully_connected(net, 4, weights_init = norm_init_with_seed) 
    net = tflearn.fully_connected(net, 4, weights_init = norm_init_with_seed) 
    net = tflearn.fully_connected(net, 4, weights_init = norm_init_with_seed) 
    #net = tflearn.fully_connected(net, 8, restore='False')
    net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
    net = tflearn.regression(net)

    model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

    # Train Karos Model
    model.fit(train_x, train_y, n_epoch=30, batch_size=10, show_metric=True,validation_set=0.1)
    
    # Save Karos Model
    model.save(MODEL_PATH)

    # Start TensorFlow Session
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)

    # Import as TF Model 
    builder = tf.saved_model.builder.SavedModelBuilder(MODEL_PATH)
    numx = len(train_x[0])
    numy = len(train_y[0])


    # Export for Serving
    serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
    print(serialized_tf_example)
    feature_configs = {'x': tf.FixedLenFeature(shape=[numx], dtype=tf.int64),}
    print(feature_configs)
    tf_example = tf.parse_example(serialized_tf_example, feature_configs)
    print(tf_example)
    x = tf.identity(tf_example['x'], name='x') 
    print(x)
    

    y = model.predict([train_x[0]])
    print(y)


    tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(tf.convert_to_tensor(y))

    prediction_signature = (
      tf.saved_model.signature_def_utils.build_signature_def(
          inputs={'query': tensor_info_x},
          outputs={'classes': tensor_info_y},
          method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    builder.add_meta_graph_and_variables(
      sess, [tf.saved_model.tag_constants.SERVING],
      signature_def_map={
          'predict_class':
              prediction_signature
      },
      main_op=tf.tables_initializer(),
      strip_default_attrs=True)

    builder.save()

     


    
