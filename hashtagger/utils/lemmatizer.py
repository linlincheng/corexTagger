import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

"""taken from https://medium.com/@gaurav5430/using-nltk-for-lemmatizing-sentences-c1bfff963258"""

lemmatizer = WordNetLemmatizer()


def check_nltk_data(
                    source_dict={'averaged_perceptron_tagger': 'taggersaveraged_perceptron_tagger',
                                 'punkt': 'tokenizers/punkt'}):
    for key, value in source_dict.items():
        try:
            nltk.data.find(value)
        except LookupError:
            nltk.download(key)
        finally:
            import nltk


# function to convert nltk tag to wordnet tag
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize_sentence(sentence):
    #  tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    #  tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #  if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:
            #  else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)
