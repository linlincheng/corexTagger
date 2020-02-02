import logging
import scipy
from hashtagger.dataProcessor import dataProcessor

logging.basicConfig(
    format='%(asctime)s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%d-%m-%Y:%H:%M:%S',
    level=logging.INFO
    )
log = logging.getLogger('test_dataProcessor')


def get_DataProcessor():
    DataProcessor = dataProcessor(data_path='./test/test_data.csv', response_field='content')
    DataProcessor.get_text_data()
    doc_words = DataProcessor.doc_words
    vocabulary = DataProcessor.vocabulary
    return(doc_words, vocabulary)


def test_dataProcessor():
    log.info('testing dataProcessor returns proper doc_words and vocabulary')
    doc_words, vocabulary = get_DataProcessor()
    assert isinstance(doc_words, scipy.sparse.csr.csr_matrix), 'doc_words type incorrect...'
    assert doc_words.shape == (3, 7), 'doc_words shape incorrect...'
    assert vocabulary == ['car', 'cat', 'catch', 'dog', 'flight', 'rabbit', 'train'], \
        'vocabulary incorrect...'
