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
from keras.layers.core import Lambda
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
import cPickle
import functools
from itertools import product
from keras.callbacks import EarlyStopping

LABELS = ['agree', 'disagree', 'discuss', 'unrelated']

train_stances_f = 'data/train_stances.csv'
train_body_f = 'data/train_bodies.csv'
test_stances_f = 'data/competition_test_stances.csv'
test_body_f = 'data/competition_test_bodies.csv'

#weighted categorical cross entropy loss function
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
ncce = functools.partial(w_categorical_crossentropy, weights=w_array)
ncce.__name__ ='w_categorical_crossentropy'

def read_binary_file(path):
    f = file(path, 'rb')
    data = cPickle.load(f)
    f.close()
    return data

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    #print string
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
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

def cnn_model(embedding_weights, train_dat, test_dat):
    max_len_h = 42
    max_len_b = 5208
    dropout = 0.5

    train_x_stn, train_x_body, train_y = train_dat
    test_x_stn, test_x_body, test_y = test_dat

    train_x_stn = np.array(sequence.pad_sequences(train_x_stn, maxlen=max_len_h))
    train_x_body = np.array(sequence.pad_sequences(train_x_body, maxlen=max_len_b))
    test_x_stn = np.array(sequence.pad_sequences(test_x_stn, maxlen=max_len_h))
    test_x_body = np.array(sequence.pad_sequences(test_x_body, maxlen=max_len_b))



    body_text_layer = Input(shape=(max_len_b,), dtype='int32', name="body_input")
    embedded_layer_body = Embedding(embedding_weights.shape[0], embedding_weights.shape[1], mask_zero=False,
                                    input_length=max_len_b, weights=[embedding_weights], trainable=False)(body_text_layer)

    conv1 = Conv1D(filters=128, kernel_size=1, padding='same', activation='relu')
    conv2 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')
    conv3 = Conv1D(filters=128, kernel_size=5, padding='same', activation='relu')
    conv4 = Conv1D(filters=128, kernel_size=7, padding='same', activation='relu')
    conv5 = Conv1D(filters=128, kernel_size=9, padding='same', activation='relu')
    conv6 = Conv1D(filters=128, kernel_size=11, padding='same', activation='relu')
    conv7 = Conv1D(filters=128, kernel_size=13, padding='same', activation='relu')
    conv8 = Conv1D(filters=128, kernel_size=15, padding='same', activation='relu')

    conv1a = conv1(embedded_layer_body)
    glob1a_b = GlobalAveragePooling1D()(conv1a)
    conv2a = conv2(embedded_layer_body)
    glob2a_b = GlobalAveragePooling1D()(conv2a)
    conv3a = conv3(embedded_layer_body)
    glob3a_b = GlobalAveragePooling1D()(conv3a)
    conv4a = conv4(embedded_layer_body)
    glob4a_b = GlobalAveragePooling1D()(conv4a)
    conv5a = conv5(embedded_layer_body)
    glob5a_b = GlobalAveragePooling1D()(conv5a)
    conv6a = conv6(embedded_layer_body)
    glob6a_b = GlobalAveragePooling1D()(conv6a)
    conv7a = conv7(embedded_layer_body)
    glob7a_b = GlobalAveragePooling1D()(conv7a)
    conv8a = conv8(embedded_layer_body)
    glob8a_b = GlobalAveragePooling1D()(conv8a)

    merge_body = concatenate([glob1a_b, glob2a_b, glob3a_b, glob4a_b, glob5a_b, glob6a_b, glob7a_b, glob8a_b])

    stance_text_layer = Input(shape=(max_len_h,), dtype='int32', name="stance_input")
    embedded_layer_stance = Embedding(embedding_weights.shape[0], embedding_weights.shape[1], mask_zero=False,
                                    input_length=max_len_h, weights=[embedding_weights], trainable=False)(stance_text_layer)


    conv1_stn = Conv1D(filters=128, kernel_size=1, padding='same', activation='relu')
    conv2_stn = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')
    conv3_stn = Conv1D(filters=128, kernel_size=5, padding='same', activation='relu')

    conv1a_s = conv1_stn(embedded_layer_stance)
    glob1a_s = GlobalAveragePooling1D()(conv1a_s)
    conv2a_s = conv2_stn(embedded_layer_stance)
    glob2a_s = GlobalAveragePooling1D()(conv2a_s)
    conv3a_s = conv3_stn(embedded_layer_stance)
    glob3a_s = GlobalAveragePooling1D()(conv3a_s)

    merge_stn = concatenate([glob1a_s, glob2a_s, glob3a_s])

    

    head_body_concat = concatenate([merge_body, merge_stn])
	
    hidden_layer = Dense(600, activation='relu', kernel_initializer="glorot_uniform")(head_body_concat)
    dropout_hidden = Dropout(dropout)(hidden_layer)

    hidden_layer_2 = Dense(500, activation='relu', kernel_initializer="glorot_uniform")(hidden_layer)
    dropout_hidden_2 = Dropout(dropout/2)(hidden_layer_2)	

    output_layer = Dense(4, activation='softmax')(dropout_hidden_2)

    model = Model([body_text_layer, stance_text_layer], output=output_layer)

    adam = Adam(lr=0.001)

    #model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.compile(loss=ncce, optimizer=adam, metrics=['accuracy'])
    earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, \
                          verbose=1, mode='auto')
    callbacks_list = [earlystop]
    model.fit([train_x_body, train_x_stn], train_y, batch_size=128, epochs=30,
              verbose=1, shuffle=True, callbacks=callbacks_list, validation_split=0.1)
    test_predictions = model.predict([test_x_body, test_x_stn], verbose=False)
    test_y = [np.argmax(pred) for pred in test_y]
    test_pred = [np.argmax(pred) for pred in test_predictions]

    print accuracy_score(test_y, test_pred)

    return test_pred



def read_data(stance_f,body_f):
    data = pd.read_csv(stance_f)
    headline_text = np.unique(np.array(data['Headline']))

    body_dat = pd.read_csv(body_f)
    body_id = np.array(body_dat['Body ID'])
    body_text = np.array(body_dat['articleBody'])
    body_dict = dict(zip(body_id,body_text))
    combined = np.array(data)

    return headline_text, body_dict, combined

def train_cnn():
    word2idx = np.load('vocab/word2idx.npy').item()
    idx2word = np.load('vocab/idx2word.npy').item()
    word_index_to_embeddings_map = read_binary_file('vocab/index_to_vec.bin')

    n_symbols = len(word2idx) + 1  # adding 1 to account for masking
    embedding_weights = np.zeros((n_symbols, 300))
    for word, index in word2idx.items():
        embedding_weights[index, :] = word_index_to_embeddings_map[word]

    train_h, train_b_dict, train_all = read_data(train_stances_f,train_body_f)
    test_h, test_b_dict, test_all = read_data(test_stances_f,test_body_f)

    train_x_stn = [clean_str(sent) for sent in train_all[:,0]]
    train_x_stn = [[word2idx[word] for word in sent] for sent in train_x_stn]

    train_x_b = [clean_str(train_b_dict[body_id]) for body_id in train_all[:,1]]
    train_x_b = [[word2idx[word] for word in sent] for sent in train_x_b]

    print train_all[:,2][0:10]
    train_y = np.array(pd.get_dummies(train_all[:,2]))
    print train_y[0:10]


    test_x_stn = [clean_str(sent) for sent in test_all[:,0]]
    test_x_stn = [[word2idx[word] for word in sent] for sent in test_x_stn]

    test_x_b = [clean_str(test_b_dict[body_id]) for body_id in test_all[:,1]]
    test_x_b = [[word2idx[word] for word in sent] for sent in test_x_b]

    print test_all[:,2][0:10]
    test_y = np.array(pd.get_dummies(test_all[:,2]))
    print test_y[0:10]

    train = train_x_stn, train_x_b, train_y
    test = test_x_stn, test_x_b, test_y

    test_preds = cnn_model(embedding_weights, train, test)
    test_preds = [LABELS[pred] for pred in test_preds]

    test_preds = pd.DataFrame(test_preds)

    test_preds.to_csv('test_preds.csv',index=False)


if __name__ == '__main__':
    train_cnn()

