import logging
import os
import corextopic
import scipy
from hashtagger.modelPredictor import modelPredictor


logging.basicConfig(
    format='%(asctime)s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%d-%m-%Y:%H:%M:%S',
    level=logging.INFO
    )
log = logging.getLogger('test_modelPredictor')


ModelPredictor = modelPredictor(
    text_data='cat and dogs',
    model_directory='./test/load_model'
    )
predicted_tags = ModelPredictor.predict_tags()


def test_ModelPredictor_load_model_objects():
    log.info('testing ModelPredictor properly loads model objects')
    assert isinstance(ModelPredictor.topic_model, corextopic.corextopic.Corex), \
        'topic_model model type incorrect...'
    assert isinstance(ModelPredictor.doc_words, scipy.sparse.csr.csr_matrix), \
        'topic_model doc_words type incorrect...'
    assert isinstance(ModelPredictor.vocabulary, list), \
        'topic_model vocabulary type incorrect...'


def test_modelPredictor_tags():
    log.info('testing modelPredictor returns expected tags')
    log.info('Found predicted_tags: {}'.format(predicted_tags))
    assert predicted_tags[0] == ['animals'], 'expected predicted tags incorrect...'
