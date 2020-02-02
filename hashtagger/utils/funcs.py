# your useful helper collection
from sklearn.feature_extraction.text import CountVectorizer
from hashtagger.utils.lemmatizer import lemmatize_sentence
import scipy.sparse as ss
import string
import logging
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import json
from collections import OrderedDict

logging.basicConfig(
    format='%(asctime)s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%d-%m-%Y:%H:%M:%S',
    level=logging.INFO
    )
log = logging.getLogger('utils.funcs')


# clean text out of irregularities
def clean_text(data_list):  # make sure to force string as list when running one time pred
    log.info('running text cleaning...')
    tokenizer = RegexpTokenizer(r'\w+')
    clean_data_list = []
    # set lemmatizer
    lemmatizer = WordNetLemmatizer()
    for text in data_list:
        # lemmatize, lower case and remove digits
        #  remove punctuations and tokenize
        text = text.translate(str.maketrans('', '', string.punctuation)).lower()
        processed_text = lemmatize_sentence(sentence=text)
        clean_data_list.append(processed_text)
    return(clean_data_list)


# vectorize your data and create a sparse matrix
def sparse_hot_encoder(data_list, vocabulary=None):
    log.info('running sparse matrix tranformation...')
    # calls select text response, and return a list
    vectorizer = CountVectorizer(stop_words='english', max_features=20000, binary=True,
                                 vocabulary=vocabulary)
    doc_words = vectorizer.fit_transform(data_list)
    #  convert to sparse matrix
    doc_words = ss.csr_matrix(doc_words)
    log.info('doc_word shape: {}'.format(doc_words.shape))
    #  get known vocabulary list
    vocabulary = list(np.asarray(vectorizer.get_feature_names()))
    return doc_words, vocabulary


# import anchor words as ordereddict
def set_anchor_words(anchor_path):
    "import and parse anchor words"
    with open(anchor_path) as handle:
        anchor_dict = json.load(handle, object_pairs_hook=OrderedDict)
    return(anchor_dict)
