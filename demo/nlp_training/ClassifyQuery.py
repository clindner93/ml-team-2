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
#from tf.keras.models import load_model


TF_CHECKPOINT_DIR = os.getenv("TF_CHECKPOINT_DIR", None)

#PATH FOR TRAINING DATA

#TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", "/mnt/traindata")
TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", ".")
TRAIN_DATA_FILE = os.getenv("TRAIN_DATA_FILE", "TrainingDataset.json")


#PATH FOR TEST DATA

#TEST_DATA_PATH = os.getenv("TEST_DATA_PATH", "/mnt/testdata")
TEST_DATA_PATH = os.getenv("TEST_DATA_PATH", ".")
TEST_DATA_FILE = os.getenv("TEST_DATA_FILE", "TestDataset.txt")

#PATH FOR TEST DATA

#TEST_CLASSES_PATH = os.getenv("TEST_CLASSES_PATH", "/mnt/testdata")
TEST_CLASSES_PATH = os.getenv("TEST_CLASSES_PATH", ".")
TEST_CLASSES_FILE = os.getenv("TEST_CLASSES_FILE", "TestDataClasses.txt")
#PATH FOR MODEL

#TF_MODEL_EXPORT_PATH = os.getenv("TF_MODEL_EXPORT_PATH", "/mnt/export")
#TF_MODEL_EXPORT_FILE = os.getenv("TF_MODEL_EXPORT_FILE ","model.tflearn")
TF_MODEL_EXPORT_PATH = os.getenv("TF_MODEL_EXPORT_PATH", "./model_path")
TF_MODEL_EXPORT_FILE = os.getenv("TF_MODEL_EXPORT_FILE","model_trial")

#PATH FOR BAG OF WORDS

#BOWORDS_PATH = os.getenv("BOWORDS_PATH", "/mnt/testdata")
BOWORDS_PATH = os.getenv("BOWORDS_PATH", ".")
BOWORDS_FILE = os.getenv("BOWORDS__FILE", "global_words.txt")

stop_words = stopwords.words('english')
nltk.download('stopwords')
nltk.download('punkt')
stemmer = SnowballStemmer("english", ignore_stopwords=True)
tf.logging.set_verbosity(tf.logging.ERROR)

'''

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
'''

def get_tf_record(sentence):
    words = sorted(list(set(open(BOWORDS_PATH +'/'+BOWORDS_FILE, 'r'))))
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words if not word in stop_words]
    bow = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bow[i] = 1
    return(np.array(bow))


def classifyQuery(query=None):
    #print(" Logging for Classification entry")
    #tf.reset_default_graph() 
    #model.load('./model.tflearn',scope_for_restore="scope1", weights_only=True, create_new_session=False)
    #model.load(TF_MODEL_EXPORT_DIR+'/'+TF_MODEL_FILE_PATH)
    # model = keras.models.load_model(TF_MODEL_EXPORT_PATH+'/'+TF_MODEL_EXPORT_FILE)
    pass
	    
    '''
    prediction = saved_model.predict(tf_input_predict)[0]
    print(prediction)
    if prediction[np.argmax(saved_model.predict([get_tf_record(query)]))] >= 0.85:
    #print (classes[np.argmax(prediction)])
        return (classes[np.argmax(prediction)])
    '''

if __name__ == '__main__':
    success = 0
    query_file = open(TEST_DATA_PATH +'/'+TEST_DATA_FILE, 'r') 
    class_file = open(TEST_CLASSES_PATH+'/'+TEST_CLASSES_FILE,'r')
    indx = 0
    classified_classes = ['Unknown'] * 16


    path = TF_MODEL_EXPORT_PATH+'/'+TF_MODEL_EXPORT_FILE

    with tf.Session(graph=tf.Graph()) as sess:

        tf.saved_model.loader.load(sess, set([ tf.saved_model.tag_constants.SERVING ]), path)

        for indx, query in enumerate(query_file):
            input_for_model = get_tf_record(query)
            print (input_for_model)

            tf_input_predict = tf.convert_to_tensor(input_for_model, dtype=tf.int64)
            prediction = sess.run('query:0',feed_dict=tf_input_predict)

            print (prediction)




    #print('First entry is' + classified_classes[0])
    #print('Last entry is' + classified_classes[15])

    
    #print(classified_classes)
    #print(len(classified_classes))
    #print("\n\n\n")
    for indx, query in enumerate(query_file):
        classified_classes[indx] = classifyQuery(query)
        #print indx

       
        
    #print classified_classes
    test_class =  list(class_file)
    test_class = list(map(lambda s: s.strip('\n'), test_class))
    #print(test_class)
    

    for ind in range(0,16):
        #print (test_class[ind])
        #print (classified_classes[ind])
        if str(classified_classes[ind]) == str(test_class[ind]):
            success +=1 
            #print success

    perc_success = float(success*100/16)
    print("Model Accuracy Calculated Using Test Dataset = " + str(perc_success))
