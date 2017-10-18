import pandas as pd
import numpy as np
import re
from collections import defaultdict
from gensim.models.keyedvectors import KeyedVectors
import cPickle

train_stances_f = 'data/train_stances.csv'
train_body_f = 'data/train_bodies.csv'
test_stances_f = 'data/competition_test_stances.csv'
test_body_f = 'data/competition_test_bodies.csv'

embeddings_file = '/home/local/UKP/chaudhuri/QuoraKaggle/GoogleNews-vectors-negative300.bin'

def read_data(stance_f,body_f):
    data = pd.read_csv(stance_f)
    headline_text = np.unique(np.array(data['Headline']))

    body_dat = pd.read_csv(body_f)
    body_id = np.array(body_dat['Body ID'])
    body_text = np.array(body_dat['articleBody'])
    body_dict = dict(zip(body_id,body_text))
    combined = np.array(data)

    return headline_text, body_dict, combined

def get_index_to_embeddings_mapping(vocab, word_vecs):
    embeddings = {}
    for word in vocab.keys():
        try:
            embeddings[word] = word_vecs[word]
        except KeyError:
            # map index to small random vector
            # print "No embedding for word '"  + word + "'"
            embeddings[word] = np.random.uniform(-0.01, 0.01, 300)
    return embeddings

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

    #return string.lower().strip().split()
    return string.lower().strip()

def create_vocab():

    train_h, train_b_dict, train_all = read_data(train_stances_f,train_body_f)
    test_h, test_b_dict, test_all = read_data(test_stances_f,test_body_f)

    print train_h.shape
    train_h = [clean_str(sent) for sent in train_h]
    test_h = [clean_str(sent) for sent in test_h]

    #train_body = [clean_str(sent) for sent in train_b_dict.values()]
    #test_body = [clean_str(sent) for sent in test_b_dict.values()]
    train_body = [(doc.split('.')) for doc in train_b_dict.values()]
    test_body = [(doc.split('.')) for doc in test_b_dict.values()]

    train_body = [[clean_str(sent) for sent in doc] for doc in train_body]
    test_body = [[clean_str(sent) for sent in doc] for doc in test_body]
    
    vocab = defaultdict(float)

    all_h = np.concatenate((train_h,test_h),axis=0)
    all_bodies = np.concatenate((train_body,test_body),axis=0)

    max_len_h = np.max([len(sent) for sent in all_h])
    max_len_body = np.max([len(sent) for sent in all_bodies])

    print "Maximum sentence length in stances:" + str(max_len_h)

    print "Maximum document length in bodies:" + str(max_len_body)

    for sent in all_h:
        for word in sent:
            vocab[word] += 1

    for sent in all_bodies:
        for word in sent:
            vocab[word] += 1

    word2idx = dict(zip(vocab.keys(), range(0, len(vocab))))
    idx2word = {v: k for k, v in word2idx.iteritems()}

    np.save('vocab/word2idx', word2idx)
    np.save('vocab/idx2word', idx2word)

    word_vecs = KeyedVectors.load_word2vec_format(embeddings_file, binary=True)
    index_to_vector_map = get_index_to_embeddings_mapping(word2idx, word_vecs)
    file_name = file('vocab/index_to_vec.bin', 'wb')
    cPickle.dump(index_to_vector_map, file_name)
    file_name.close()

if __name__ == '__main__':
    create_vocab()
