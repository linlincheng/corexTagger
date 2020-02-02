import logging
import os
import corextopic
import pytest
from hashtagger.dataProcessor import dataProcessor
from hashtagger.modelTrainer import semiSupervisedTrainer


logging.basicConfig(
    format='%(asctime)s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%d-%m-%Y:%H:%M:%S',
    level=logging.INFO
    )
log = logging.getLogger('test_semiSupervisedTrainer')

@pytest.fixture
def DataProcessor():
    DataProcessor = dataProcessor(data_path='./test/test_data.csv', response_field='content')
    DataProcessor.get_text_data()
    return(DataProcessor)

@pytest.fixture
def SemiSupervisedTrainer(DataProcessor):
    SemiSupervisedTrainer = semiSupervisedTrainer(
        words=DataProcessor.vocabulary,
        doc_words=DataProcessor.doc_words,
        n_topic=3,
        save_model=True,
        model_directory='./test/model/',
        print_words=True,
        anchor_path='./test/test_anchor_words.json')
    SemiSupervisedTrainer.train_model()
    SemiSupervisedTrainer.save_model_object()
    return(SemiSupervisedTrainer)


def test_semiSupervisedTrainer_train_model(SemiSupervisedTrainer):
    log.info('testing semiSupervisedTrainer returns proper doc_words and vocabulary')
    assert isinstance(SemiSupervisedTrainer.topic_model, corextopic.corextopic.Corex), \
        'topic_model object type incorrect...'


def test_semiSupervisedTrainer_save_model_object(SemiSupervisedTrainer):
    log.info('testing semiSupervisedTrainer saves model objects properly')
    file_list_set = set(os.listdir(SemiSupervisedTrainer.model_object_path))
    print(os.listdir(SemiSupervisedTrainer.model_object_path))
    expected_list_set = set(['model', 'words', 'anchor_words'])
    assert set(expected_list_set).issubset(file_list_set), 'save_model_objects not working properly, \
                                                           one or two files missing...'


@pytest.fixture(scope="module")
def remove_dir(model_dir='./test/model'):
    yield
    print('Deleting test model files...')
    for root, dirs, files in os.walk(model_dir, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
