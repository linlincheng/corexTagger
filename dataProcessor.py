import pandas as pd
import logging
from corextopic import corextopic as ct
from corextopic import vis_topic as vt
import pickle
import numpy as numpy
#from nltk.tokenize import RegexpTokenizer
from utils.funcs import clean_text, sparse_hot_encoder

logging.basicConfig(
    format='%(asctime)s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%d-%m-%Y:%H:%M:%S',
    level=logging.INFO
    )
log = logging.getLogger('dataProcessor')


class dataProcessor():
    """import and preprocess your text data for topic modeling"""

    def __init__(self, data_path='./airline.csv', response_field='content'):
        log.info('Initializating dataProcessor...')
        self.data_path = data_path
        self.response_field = response_field
        self.vocabulary = None
        self.data_frame = None
        self.doc_words = None

    def _import_data(self):
        return(pd.read_csv(self.data_path))

    def _check_response_field(self, data):
        if self.response_field is not in list(data):
            raise NameError('response_field not found...')

    def select_text_response(self):
        self.data_frame = self._import_data()
        # check response_field setup
        self._check_response_field()
        text_list = data[self.response_field]
        return(text_list)

    def get_text_data(self, ):
        # import data and select text field
        text_list = self.select_text_response()
        # preprocess text
        clean_text_list = clean_text(data_dict=text_list)
        # get sparse matrix
        self.doc_words = sparse_hot_encoder(clean_text_list, vocabulary=self.vocabulary)

    def get_vocab(self, vectorizer):
        self.vocabulary = list(np.asarray(vectorizer.get_feature_names()))
