import pandas as pd
import logging
from corextopic import corextopic as ct
from corextopic import vis_topic as vt
import pickle
#from nltk.tokenize import RegexpTokenizer
from hashtagger.utils.funcs import clean_text, sparse_hot_encoder

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
        self.doc_words = None

    @property
    def data_frame(self):
        return(pd.read_csv(self.data_path))

    def _check_response_field(self, data):
        if self.response_field not in list(data):
            raise NameError('response_field not found...')

    def select_text_response(self):
        data = self.data_frame
        # check response_field setup
        self._check_response_field(data=data)
        text_list = data[self.response_field]
        return(text_list)

    def get_text_data(self):
        log.info('Selecting text field...')
        # import data and select text field
        text_list = self.select_text_response()
        # preprocess text
        clean_text_list = clean_text(data_list=text_list)
        # get sparse matrix
        self.doc_words, self.vocabulary = sparse_hot_encoder(
                                                clean_text_list,
                                                vocabulary=self.vocabulary
                                                    )
