import pandas as pd
from corextopic import corextopic as ct
from corextopic import vis_topic as vt
import pickle
import logging
import os
from datetime import datetime
import logging
from abc import ABC, abstractmethod
from hashtagger.utils.funcs import set_anchor_words


logging.basicConfig(
    format='%(asctime)s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%d-%m-%Y:%H:%M:%S',
    level=logging.INFO
    )
log = logging.getLogger('modelTrainer')


class modelTrainer(ABC):
    "base class to unify model training for unsupervised and semisupervised scenarios"

    def __init__(self, words, doc_words,
                 n_topic=50, save_model=True,
                 model_directory='model/', print_words=True):
        self.words = words
        self.doc_words = doc_words
        self.n_topic = n_topic
        self.print_words = print_words
        self.save_model = save_model
        self.model_directory = model_directory
        self.datetime = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        self.topic_model = None
        #self.file_path = None

    @abstractmethod
    def train_model(self):
        return

    # Print all topics from the CorEx topic model
    def print_topic_words(self, topic_model):
        log.info('Printing topic words: ')
        topics = topic_model.get_topics()
        for n, topic in enumerate(topics):
            topic_words, _ = zip(*topic)
            print('{}: '.format(n) + ','.join(topic_words))

    # create file path
    @property
    def model_object_path(self):
        model_object_path = self.model_directory+self.datetime
        # check and create model directory
        if not os.path.exists(model_object_path):
            os.makedirs(model_object_path)
        # add file path as class attribute
        return(model_object_path)

    # save model words bundle
    def save_model_object(self):
        log.info('Saving model objects...')
        # save model
        full_name = self.model_object_path+'/model'
        self.topic_model.save(full_name)
        # save words
        pickle.dump(self.words, open(self.model_object_path+"/words", 'wb'), protocol=-1)


class unSupervisedTrainer(modelTrainer):
    """train your corex model"""
    def __init__(self, words, doc_words,
                 n_topic=50, save_model=True,
                 model_directory='model/', print_words=True):
        log.info('Initializing unSupervisedTrainer class...')
        modelTrainer.__init__(
            self, words=words, doc_words=doc_words,
            n_topic=n_topic, save_model=save_model,
            model_directory=model_directory, print_words=print_words
            )

    # Train model
    def train_model(self):
        log.info('Running model training...')
        # Train the CorEx topic model with 50 topics
        topic_model = ct.Corex(n_hidden=self.n_topic, words=self.words, max_iter=200, verbose=False, seed=1)
        topic_model.fit(self.doc_words, words=self.words)
        # save to class
        self.topic_model = topic_model
        if self.print_words:
            self.print_topic_words(topic_model=topic_model)


class semiSupervisedTrainer(modelTrainer):
    """train your wicked semi supervised corex model"""
    def __init__(
        self, words, doc_words,
        n_topic=50, save_model=True,
        model_directory='model/', print_words=True,
        anchor_strength=6, anchor_path='./anchor_words.json'
                 ):
        log.info('Initializing semiSupervisedTrainer class')
        modelTrainer.__init__(
            self, words=words, doc_words=doc_words,
            n_topic=n_topic, save_model=save_model,
            model_directory=model_directory, print_words=print_words)
        self.anchor_strength = anchor_strength
        self.anchor_path = anchor_path
        self.anchor_dict = None

    def train_model(self):
        log.info('Running model training...')
        """ Train semisupervised topic model with n topics"""
        # set anchor words
        self.anchor_dict = set_anchor_words(anchor_path=self.anchor_path)
        anchor_words = list(self.anchor_dict.values())
        # train model
        topic_model = ct.Corex(n_hidden=self.n_topic, words=self.words, max_iter=200, verbose=False, seed=1)
        topic_model.fit(self.doc_words, words=self.words, anchors=anchor_words, anchor_strength=self.anchor_strength)
        # save to class
        self.topic_model = topic_model
        if self.print_words:
            self.print_topic_words(topic_model=topic_model)

    def save_model_object(self):
        # save model and words
        super().save_model_object()
        # save copy of anchor words used
        log.info('Saving copy of anchor words...')
        pickle.dump(self.anchor_dict, open(super().model_object_path+"/anchor_words", 'wb'), protocol=-1)
