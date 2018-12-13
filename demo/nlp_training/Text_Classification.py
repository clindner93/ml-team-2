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

#print(tf.__version__)
#print(np.__version__)

#PATH FOR CHEKCPOINT

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


nltk.download('stopwords')
nltk.download('punkt')
stemmer = SnowballStemmer("english", ignore_stopwords=True)
tf.logging.set_verbosity(tf.logging.ERROR)

with open(TRAIN_DATA_PATH +'/'+TRAIN_DATA_FILE) as json_data:
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



#print ("\n\n\n")
nltk.download('stopwords')
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
    #print(token_words)
    for w in words:
        bow.append(1) if w in token_words else bow.append(0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    #print(bow)
    #print(output_row)
    training.append([bow, output_row])


with open(BOWORDS_PATH+'/'+BOWORDS_FILE, 'w') as f:
    for word in words:
        f.write("%s\n" % word)

random.shuffle(training)

training = np.array(training)

train_x = list(training[:, 0])
print(len(train_x[0]))

train_y = list(training[:, 1])

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


#Getting input sentence and parsing


def get_tf_record(sentence):
    global words
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
    #model.load(TF_MODEL_EXPORT_PATH+'/'+TF_MODEL_EXPORT_FILE)
    model.load(TF_MODEL_EXPORT_PATH)

    #print(query)
    prediction = model.predict([get_tf_record(query)])[0]
    #print(prediction)
    if prediction[np.argmax(model.predict([get_tf_record(query)]))] >= 0.85:
        #print (classes[np.argmax(prediction)])
        return (classes[np.argmax(prediction)])
        
    '''
    else:
		print('The query cannot be assigned to any class')
        #return_str = Unknown
        #return (return_str)
    '''
def trainData():


    model.fit(train_x, train_y, n_epoch=2, batch_size=10, show_metric=True,validation_set=0.05)
    #model.save('model.tflearn')
    model.save(TF_MODEL_EXPORT_PATH+'/'+TF_MODEL_EXPORT_FILE)

#############################  NEW PART ATTEMPT

    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    builder = tf.saved_model.builder.SavedModelBuilder(TF_MODEL_EXPORT_PATH+'/'+TF_MODEL_EXPORT_FILE)
    numx =len(train_x[0])
    numy = 3


    serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
    print(serialized_tf_example)
    feature_configs = {'x': tf.FixedLenFeature(shape=[len(train_x[0])], dtype=tf.int64),}
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



    #tf_example = tf.parse_example(serialized_tf_example, feature_configs)
    #x = tf.identity(tf_example['x'], name='x')  # use tf.identity() to assign name
    
    #tensor_info_x = tf.saved_model.utils.build_tensor_info(train_x[0])
    #tensor_info_y = tf.saved_model.utils.build_tensor_info(train_y[0])



    #model.save(TF_MODEL_EXPORT_PATH)
    return "Model trained successfully"
#Uncomment below line when you need to train and generate a model
#Comment trainData() if using trained model





trainData()

if __name__ == '__main__':
    success = 0
    query_file = open(TEST_DATA_PATH +'/'+TEST_DATA_FILE, 'r') 
    class_file = open(TEST_CLASSES_PATH+'/'+TEST_CLASSES_FILE,'r')
    indx = 0
    classified_classes = ['Unknown'] * 16

    #print('First entry is' + classified_classes[0])
    #print('Last entry is' + classified_classes[15])

    
    #print(classified_classes)
    #print(len(classified_classes))
    #print("\n\n\n")
    '''
    COMMENTING THIS PART TO NOT INVOKE CLASSIFY
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

    '''

     


    