import pandas as pd
import logging
import scipy
from hashtagger.utils.funcs import clean_text, sparse_hot_encoder

logging.basicConfig(
    format='%(asctime)s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%d-%m-%Y:%H:%M:%S',
    level=logging.INFO
    )
log = logging.getLogger('test_utils_funcs')


test_data = pd.read_csv('test/test_data.csv')
clean_data = clean_text(test_data['content'])
doc_words, vocabulary = sparse_hot_encoder(data_list=clean_data)


def test_clean_text_lower_case():
    log.info('Testing clean_text returns all lower cases...')
    assert clean_data[1] == 'car train flight', 'not returning proper lower cases: {}'.\
                            format(clean_data[1])


def test_clean_text_lemmatizer():
    log.info('Testing clean_text returns properly lemmatized format...')
    assert clean_data[2] == 'the cat catch two train', 'not returning properly lematized format: {}'.\
                            format(clean_data[2])


def test_sparse_hot_encoder_doc_words():
    log.info('Testing sparse hot encoder returns sparse doc_words properly...')
    assert isinstance(doc_words, scipy.sparse.csr.csr_matrix), 'doc_words type incorrect...'
    assert doc_words.shape == (3, 7), 'doc_words shape incorrect...'


def test_sparse_hot_encoder_vocabulary():
    log.info('Testing sparse hot encoder returns vocabulary properly...')
    assert vocabulary == ['car', 'cat', 'catch', 'dog', 'flight', 'rabbit', 'train'], \
        'vocabulary incorrect...'
