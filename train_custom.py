import pandas as pd
import numpy as np
import re
from collections import defaultdict
from keras.engine import Input
from keras.engine import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Conv1D, GlobalAveragePooling1D
from keras.layers.merge import Concatenate, Add, concatenate
from keras.preprocessing import sequence
from keras import backend as K
from keras.layers.core import Lambda, Flatten
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
import cPickle
import functools
import keras
from itertools import product
from keras.callbacks import EarlyStopping
from keras.layers import merge
import gensim
from keras.layers.core import *
from gensim.models import word2vec
from keras.layers.normalization import BatchNormalization
from keras.initializers import glorot_uniform
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

#define the required variables

LABELS = ['agree', 'disagree', 'discuss', 'unrelated']

train_stances_f = 'data/train_stances.csv'
train_body_f = 'data/train_bodies.csv'
test_stances_f = 'data/competition_test_stances.csv'
test_body_f = 'data/competition_test_bodies.csv'
SINGLE_ATTENTION_VECTOR = False
#google_word_vec_file = './GoogleNews-vectors-negative300.bin'
nf_words = []
#stop = stopwords.words('english')
with open('./data/english') as f:
    stop = f.read().splitlines()

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)
    
def w_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.expand_dims(y_pred_max, 1)
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):

        final_mask += (K.cast(weights[c_t, c_p],K.floatx()) * K.cast(y_pred_max_mat[:, c_p] ,K.floatx())* K.cast(y_true[:, c_t],K.floatx()))
    return K.categorical_crossentropy(y_pred, y_true) * final_mask
w_array = np.ones((4,4))
w_array[0,3] = 1.2
w_array[1,3] = 1.2
w_array[2,3] = 1.2
#w_array[0,2] = 1.1
ncce = functools.partial(w_categorical_crossentropy, weights=w_array)
ncce.__name__ ='w_categorical_crossentropy'


def attention_3d_block(inputs, time_steps):

    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = (inputs.shape[2])
    print input_dim
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, time_steps))(a)
    a = Dense(time_steps, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul


def sentence_attn(max_no_sent):

    body_inp = Input(shape=( max_no_sent, 300, ), dtype=np.float32, name = "body_input")
    stance_inp = Input(shape=(300,), dtype=np.float32, name = "stance_input")

    print body_inp
    #attention_body = AttentionWithContext()(body_inp)
    body_i = Permute((2,1))(body_inp)
    body_i = Reshape((300, max_no_sent))(body_i)
    #attention layer over body input of shape B, I, T
    attention_probs = Dense((max_no_sent), activation='softmax', name='attention_probs')(body_i)
    attn_probs = Permute((2,1), name='attn_vec')(attention_probs)
    attention_body = merge([body_inp, attn_probs], name='attention_mul', mode='mul')
    print attention_body
    attention_mul = Lambda(lambda x: K.sum(x, axis=1))(attention_body)
    print attention_mul
    #merge_l = merge([attention_mul, stance_inp], mode='concat')
    dense_1 = Dense(2500, activation="tanh", kernel_initializer="glorot_uniform")(attention_mul)
    drop_1 = Dropout(0.8)(dense_1)
    norm_1 = BatchNormalization()(drop_1)
    #flat = Flatten()(norm_1)
    merge_1 = merge([norm_1, stance_inp], mode='concat')
    #merge_1 = concatenate([flat, stance_inp], axis=-1)

    dense_2 = Dense(1200, activation="tanh", kernel_initializer="glorot_uniform")(merge_1)
    drop_2 = Dropout(0.6)(dense_2)
    norm_2 = BatchNormalization()(drop_2)
   
    #dense_3 = Dense(700, activation="tanh", kernel_initializer="glorot_uniform")(norm_2)
    #drop_3 = Dropout(0.6)(dense_3)
    #norm_3 = BatchNormalization()(drop_3)


    out_layer = Dense(4, activation="softmax")(norm_2)
    model = Model([body_inp, stance_inp], output=[out_layer])

    adam = Adam(lr=0.0001)

    model.compile(loss=ncce, optimizer=adam, metrics=['accuracy'])

    return model


def get_vectors(word_vecs, word):

    try:
        return word_vecs[word]
    except KeyError:
        nf_words.append(word)
        return np.random.uniform(-0.25, 0.25, 300)


def get_avg_vectors(word_vecs, sentence):

    out_vec = np.zeros(300)
    for word in sentence:
        out_vec = np.add(out_vec, get_vectors(word_vecs, word))

    return out_vec


def read_data(stance_f,body_f):

    data = pd.read_csv(stance_f)
    headline_text = np.unique(np.array(data['Headline']))

    body_dat = pd.read_csv(body_f)
    body_id = np.array(body_dat['Body ID'])
    body_text = np.array(body_dat['articleBody'])
    body_dict = dict(zip(body_id,body_text))
    combined = np.array(data)

    return headline_text, body_dict, combined

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    #print string
    string = re.sub(r"[^A-Za-z0-9(),]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.lower().strip().split()


def run_model():

    #load google pre-trained wor vectors:
    #google_vecs = gensim.models.KeyedVectors.load_word2vec_format(google_word_vec_file, binary=True)
    google_vecs = word2vec.Word2Vec.load('./data/Corpus_word2vec')
    train_h, train_b_dict, train_all = read_data(train_stances_f, train_body_f)
    test_h, test_b_dict, test_all = read_data(test_stances_f, test_body_f)

    train_x_stn = [clean_str(sent) for sent in train_all[:, 0]]
    print train_x_stn[0]
    train_x_stn = np.array([get_avg_vectors(google_vecs, sent) for sent in train_x_stn])
    #train_x_stn = [[word2idx[word] for word in sent] for sent in train_x_stn]

    train_x_b = [train_b_dict[body_id].split('.') for body_id in train_all[:, 1]]
    max_no_sent = np.max([len(body) for body in train_x_b])
    print max_no_sent
    train_x_b = [[clean_str(sent) for sent in body] for body in train_x_b]
    train_x_b = [[[word for word in sent if word not in stop] for sent in body] for body in train_x_b]
    train_x_b = [[get_avg_vectors(google_vecs, sent) for sent in body] for body in train_x_b]
    print np.array(train_x_b[0]).shape
    train_x_b = np.array([np.concatenate((np.array(body_vecs), np.zeros((max_no_sent - len(body_vecs), 300)))) for body_vecs in train_x_b])
    print train_x_b[0].shape
    train_y = np.array(pd.get_dummies(train_all[:, 2]))

    #process test_data
    test_x_stn = [clean_str(sent) for sent in test_all[:, 0]]
    test_x_stn = np.array([get_avg_vectors(google_vecs, sent) for sent in test_x_stn])
    test_x_b = [test_b_dict[body_id].split('.') for body_id in test_all[:, 1]]
    test_x_b = [[clean_str(sent) for sent in body] for body in test_x_b]
    test_x_b = [[[word for word in sent if word not in stop] for sent in body] for body in test_x_b]
    test_x_b = [[get_avg_vectors(google_vecs, sent) for sent in body] for body in test_x_b]
    test_x_b = np.array([np.concatenate((np.array(body_vecs), np.zeros((max_no_sent - len(body_vecs), 300)))) for body_vecs in test_x_b])
    print test_all[:, 2][0:10]
    test_y = np.array(pd.get_dummies(test_all[:, 2]))
    print test_y[0:10]
    model = sentence_attn(max_no_sent)
    print model.summary()

    #early stopping
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3,
                          verbose=1, mode='auto')
    callbacks_list = [earlystop]
    #print np.array(zip(train_x_b, train_x_stn))

    train_x, val_x, train_y, val_y = train_test_split(zip(train_x_b,train_x_stn), train_y, test_size=0.2, random_state=666, stratify=train_y)
   
    train_x = np.array(train_x)
    val_x = np.array(val_x)
    val_y = np.array(val_y)
    train_y = np.array(train_y)
    print train_x.shape, val_x.shape, train_y.shape, val_y.shape
    train_x_b = np.array([a[0] for a in train_x])
    train_x_stn = np.array([a[1] for a in train_x])
    val_x_b = np.array([a[0] for a in val_x])
    val_x_stn = np.array([a[1] for a in val_x])
    print train_x_b.shape, train_x_stn.shape, val_x_b.shape, val_x_stn.shape

    #training the model
    model.fit([train_x_b, train_x_stn], train_y, batch_size=128, epochs=120,
              verbose=1, shuffle=True, callbacks=callbacks_list, validation_data=[[val_x_b, val_x_stn], val_y])

    test_predictions = model.predict([test_x_b, test_x_stn], verbose=False)
    test_y = [np.argmax(pred) for pred in test_y]
    test_pred = [np.argmax(pred) for pred in test_predictions]
    test_preds = [LABELS[pred] for pred in test_pred] 
    custom_result = pd.read_csv(test_stances_f)
    custom_result['Stance'] = test_preds

    custom_result.to_csv('sentence_attn_25.csv', index=False)
    print (set(nf_words))

    print accuracy_score(test_y, test_pred)


if __name__ == '__main__':
    run_model()
