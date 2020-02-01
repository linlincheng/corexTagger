import pandas as pd
from corextopic import corextopic as ct
from corextopic import vis_topic as vt
import pickle
import logging
import numpy as numpy
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

    def __init__(self, words=DataProcessor.vocabulary, doc_words=DataProcessor.doc_words,
                 n_topic=50, save_model=True, model_directory='model/', print_words=True):
        self.words = words
        self.doc_word = doc_word
        self.n_topic = n_topic
        self.print_words = print_words
        self.save_model = save_model
        self.model_directory = model_directory
        self.topic_model = None
        self.file_path = None

    @abstractmethod
    def train_model(self, print_words):
        return

    # Print all topics from the CorEx topic model
    def print_topic_words(self, topic_model):
        topics = topic_model.get_topics()
        for n, topic in enumerate(topics):
            topic_words, _ = zip(*topic)
            print('{}: '.format(n) + ','.join(topic_words))

    # save model words bundle
    def save_model_object(self):
        datetime_string = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        file_path = self.model_directory+datetime_string
        # check and create model directory
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        # add file path as class attribute
        self.file_path = file_path
        #save model
        full_name = model_directory+datetime_string+'/model'
        self.topic_model.save(full_name)
        #save words
        pickle.dump(self.words, open(file_path+"/words", 'wb'), protocol=-1)


class unSupervisedTrainer(modelTrainer):
    """train your corex model"""
    def __init__(self, words=DataProcessor.vocabulary, doc_words=DataProcessor.doc_words,
                 n_topic=50, save_model=True, model_directory='model/', print_words=True):
        modelTrainer.__init__(
            words=words, doc_words=doc_words,
            n_topic=n_topic, save_model=save_model,
            model_directory=model_directory, print_words=print_words)

    # Train model
    def train_model(self, print_words):
        # Train the CorEx topic model with 50 topics
        topic_model = ct.Corex(n_hidden=self.n_topic, words=self.words, max_iter=200, verbose=False, seed=1)
        topic_model.fit(self.doc_words, words=self.words)
        # save to class
        self.topic_model = topic_model
        if self.print_words:
            self.print_topic_words()


class semiSuperviserTrainer(modelTrainer):
    """train your wicked semi supervised corex model"""
    def __init__(self, words=DataProcessor.vocabulary, doc_words=DataProcessor.doc_words,
                 n_topic=50, save_model=True, model_directory='model/', print_words=True, 
                 anchor_strengh=6, anchor_path='../anchor_words.json'):
        modelTrainer.__init__(
            words=words, doc_words=doc_words,
            n_topic=n_topic, save_model=save_model,
            model_directory=model_directory, print_words=print_words)
        self.anchor_strength = anchor_strength
        self.anchor_path = anchor_path
        self.anchor_dict = None

    def train_model(self, print_words=True):
        """ Train semisupervised topic model with n topics"""
        # set anchor words
        self.anchor_dict = set_anchor_words(anchor_path=self.anchor_path)
        anchor_words = list(self.anchor_dict.values())
        # train model
        topic_model = ct.Corex(n_hidden=self.n_topic, words=self.words, max_iter=200, verbose=False, seed=1)
        topic_model.fit(self.doc_words, words=self.words, anchors=anchor_words, anchor_strength=self.anchor_level)
        # save to class
        self.topic_model = topic_model
        if self.print_words:
            self.print_topic_words()

    def save_model_object(self):
        # save model and words
        super().save_model_object()
        # save copy of anchor words used
        pickle.dump(self.anchor_dict, open(super().file_path+"/anchor_words", 'wb'), protocol=-1)
