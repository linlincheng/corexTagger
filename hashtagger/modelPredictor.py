import pandas as pd
import logging
import os
import numpy as np
from corextopic import corextopic as ct
from corextopic import vis_topic as vt
from hashtagger.utils.funcs import clean_text, sparse_hot_encoder, set_anchor_words

logging.basicConfig(
    format='%(asctime)s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%d-%m-%Y:%H:%M:%S',
    level=logging.INFO
    )
log = logging.getLogger('modelPredictor')


class modelPredictor():
    "modelPredictors predict your tag for you"
    def __init__(self, text_data, model_directory='./model'):
        # check text_data class
        if isinstance(text_data, str):
            text_data = [text_data]
        self.text_data = text_data
        self.model_directory = model_directory
        self.doc_words = None
        self.vocabulary = None
        self.topic_model = None
        self.anchor_dict = None

    def preprocess_text(self):
        clean_text_list = clean_text(self.text_data)
        self.doc_words, _ = sparse_hot_encoder(clean_text_list, vocabulary=self.vocabulary)

    def load_model_object(self):
        model_object_path = self.model_directory+'/'+sorted(os.listdir(self.model_directory))[-1]
        # load model
        model_file_name = model_object_path+'/model'
        self.topic_model = ct.load(filename=model_file_name)
        # load vocabulary
        self.vocabulary = ct.load(filename=model_object_path+'/words')
        # load anchor_words
        self.anchor_dict = ct.load(filename=model_object_path+'/anchor_words')

    def predict_tags(self):
        # run import and preprocessing steps
        self.load_model_object()
        self.preprocess_text()
        # predict
        corex_doc_labels = self.topic_model.transform(self.doc_words, details=False)
        # return labels
        # process doc_label
        predicted_tags = self._process_doc_labels(doc_labels=corex_doc_labels)
        #log.info('Predicted tags: {}'.format(', '.join(predicted_tags)))
        return(predicted_tags)

    def _process_doc_labels(self, doc_labels):
        anchor_length = len(self.anchor_dict)
        anchor_tags = self.anchor_dict
        # select only anchored tags
        anchor_doc_labels = doc_labels[0:anchor_length, :]
        anchor_tags_list = []
        for anchor_doc_label in anchor_doc_labels:
            anchor_tags = [tag for (tag, indicator) in
                           zip(anchor_tags, anchor_doc_label) if indicator]
            anchor_tags_list.append(anchor_tags)
        return(anchor_tags_list)
