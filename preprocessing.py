import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from gensim.models.doc2vec import TaggedDocument

#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('punkt')
#nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # remove punctuation and lowercase text
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    # remove stop words
    text = [word for word in text.split() if word not in stop_words]
    # stem and lemmatize words
    #text = [lemmatizer.lemmatize(stemmer.stem(word)) for word in text]
    return text

def preprocess_corpus(corpus):
    preprocessed_corpus = [preprocess_text(text) for text in corpus]
    return preprocessed_corpus
