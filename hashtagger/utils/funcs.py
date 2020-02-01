# your useful helper collection
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as ss
import string
import logging
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

logging.basicConfig(
    format='%(asctime)s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%d-%m-%Y:%H:%M:%S',
    level=logging.INFO
    )
log = logging.getLogger('utils.funcs')


# clean text out of irregularities
def clean_text(data_list):  # make sure to force string as list when running one time pred
    log.info('running text cleaning...')
    tokenizer = RegexpTokenizer(r'\w+')
    clean_data_list = []
    # set lemmatizer
    lemmatizer = WordNetLemmatizer()
    for text in data_list:
        #  remove punctuations and tokenize
        text.translate(str.maketrans('', '', string.punctuation))
        # lemmatize, lower case and remove digits
        text = ''.join([lemmatizer.lemmatize(word) if not word.isdigit() else
                        'NUM' for word in text])
        clean_data_list.append(text)
    return(clean_data_list)

# vectorize your data and create a sparse matrix
def sparse_hot_encoder(data_list, vocabulary=None):
    log.info('running sparse matrix tranformation...')
    # calls select text response, and return a list
    vectorizer = CountVectorizer(stop_words='english', max_features=20000, binary=True,
                                 vocabulary=vocabulary)
    doc_words = vectorizer.fit_transform(data_list)
    #  convert to sparse matrix
    doc_words = ss.csr_matrix(doc_words)
    log.info('doc_word shape: {}'.format(doc_words.shape))
    return(doc_words)

# import anchor words as ordereddict
def set_anchor_words(anchor_path):
    "import and parse anchor words"
    with open(anchor_path) as handle:
        anchor_dict = json.load(handle.read(), object_pairs_hook=OrderedDict)
    return(anchor_dict)
