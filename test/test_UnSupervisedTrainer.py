import logging
import os
import corextopic
from hashtagger.dataProcessor import dataProcessor
from hashtagger.modelTrainer import unSupervisedTrainer


logging.basicConfig(
    format='%(asctime)s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%d-%m-%Y:%H:%M:%S',
    level=logging.INFO
    )
log = logging.getLogger('test_unSupervisedTrainer')

# remove exisiting test model folders
for root, dirs, files in os.walk('./test/model', topdown=False):
    for name in files:
        os.remove(os.path.join(root, name))
    for name in dirs:
        os.rmdir(os.path.join(root, name))

"""if os.path.exists('./test/model'):
    os.rmdir('./test/model')
else:
    log.info('The model foler does not exist')
"""

DataProcessor = dataProcessor(data_path='./test/test_data.csv', response_field='content')
DataProcessor.get_text_data()


UnSupervisedTrainer = unSupervisedTrainer(
    words=DataProcessor.vocabulary,
    doc_words=DataProcessor.doc_words,
    n_topic=3, save_model=True,
    model_directory='./test/model/',
    print_words=True)
UnSupervisedTrainer.train_model()
UnSupervisedTrainer.save_model_object()


def test_unSupervisedTrainer_train_model():
    log.info('testing unSupervisedTrainer returns proper doc_words and vocabulary')
    assert isinstance(UnSupervisedTrainer.topic_model, corextopic.corextopic.Corex), \
        'topic_model object type incorrect...'


def test_unSupervisedTrainer_save_model_object():
    log.info('testing unSupervisedTrainer saves model objects properly')
    file_list_set = set(os.listdir(UnSupervisedTrainer.model_object_path))
    expected_list_set = set(['model', 'words'])
    assert set(expected_list_set).issubset(file_list_set), 'save_model_objects not working properly, \
                                                           one or two files missing...'
