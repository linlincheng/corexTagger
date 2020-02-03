## corex hashtagger - Your smarter, cooler topic modeling alternative.

Corex hashtagger sources core algorithm from [here](https://github.com/gregversteeg/corex_topic). The core is based on an information theory backed approach to empirically derive distribution of words among docs with minimum statitstical assumptions, which makes it extemely powerful when shorter texts with high volumes are concerned (less room for distribution of words and topics among documents). 

This repo showcases a potential business use case using [airline review data](https://raw.githubusercontent.com/quankiquanki/skytrax-reviews-dataset/). 

Corex hashtagger allows you to intervene in the unsupervised topic model training process by injecting anchor_words.json. For example, in airline reviews, we'd expect information to be about delays, luggages and inflight meals. All of these information can be incorporated directly in the model semi-supervised training process to further steamline the entire process. 

#### Recommened steps are: 

1) Train unsupervised model for anchor_words inspirations

2) Build your first version of anchor_words from 1), train a semisupervised version

3) Refind your anchor_words, repeat 2) until you are satisfied. 


### preprocess data: 


initialize dataProcessor class,

from hashtagger.dataProcessor import dataProcessor
DataProcessor = dataProcessor(data_path='./airline.csv', response_field='content')
DataProcessor.get_text_data()


### train unsupervised version for anchor words inspirations
from hashtagger.modelTrainer import unSupervisedTrainer
unSupervisedTrainer = unSupervisedTrainer(words=DataProcessor.vocabulary,doc_words=DataProcessor.doc_words, n_topic=20, save_model=True, model_directory='model/', print_words=True)
unSupervisedTrainer.train_model()
unSupervisedTrainer.save_model_object()

### start building up your anchor words, try out Semisupervised version 
### with anchor_words (your domain expertise) injections;
### Take printed outputs, edit your anchor_words.json, and repeat the first step
from hashtagger.modelTrainer import semiSupervisedTrainer
SemiSupervisedTrainer = semiSupervisedTrainer(words=DataProcessor.vocabulary,doc_words=DataProcessor.doc_words, n_topic=20, save_model=True, model_directory='model/', print_words=True, anchor_path='./anchor_words.json')
SemiSupervisedTrainer.train_model()
SemiSupervisedTrainer.save_model_object()

See new_playground.ipynb for more details.


To test drive the repo with appropirate env setup:

do: `test_run_interactive.sh`
to access jupyter via docker.

Note:

This is a WIP and many parts may be still under test and development. Currently only local versions work. More to come soon!


TODO:

1) implement 'Null' topic for collection of common words

2) implement bigram anchor words

3) add visualizaiton
