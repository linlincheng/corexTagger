# your useful helper collection
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as ss
import string
from nltk.stem import WordNetLemmatizer

# clean text out of irregularities
def clean_text(data_list):  # make sure to force string as list when running one time pred
    tokenizer = RegexpTokenizer(r'\w+')
    clean_data_list = []
    # set lemmatizer
    lemmatizer = WordNetLemmatizer()
    for text in data_list:
        #  remove punctuations and tokenize
        text.translate(str.maketrans('', '', string.punctuation))
        # lemmatize, lower case and remove digits
        text = ' '.join([lemmatizer.lemmatize(word) if not word.isdigit() else
                        'NUM' for word in text])
        clean_data_list.append(text)
    return(text)

# vectorize your data and create a sparse matrix
def sparse_hot_encoder(data_list, vocabulary=None):
    # calls select text response, and return a list
    vectorizer = CountVectorizer(stop_words='english', max_features=20000, binary=True,
                                 vocabulary=vocabulary)
    doc_word = vectorizer.fit_transform(data_list)
    #self.vectorizer = vectorizer
    doc_word = ss.csr_matrix(doc_word)
    log.info('doc_word shape: {}'.format(doc_word.shape))
    return(doc_words)

# import anchor words as ordereddict
def set_anchor_words(anchor_path):
    "import and parse anchor words"
    with open(anchor_path) as handle:
        anchor_dict = json.load(handle.read(), object_pairs_hook=OrderedDict)
    return(anchor_dict)
