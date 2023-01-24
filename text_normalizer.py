import re
import nltk
import spacy
import unicodedata

from bs4 import BeautifulSoup
from contractions import CONTRACTION_MAP
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import PorterStemmer

porter = PorterStemmer()
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
nlp = spacy.load('en_core_web_sm')


def remove_html_tags(text):

    pattern = re.compile('<.*?>') 

    text = re.sub(pattern, '', text)

    return text


def stem_text(text):

    text = tokenizer.tokenize(text)

    text = [porter.stem(token) for token in text]

    text = ' '.join(token for token in text)

    return text


def lemmatize_text(text):

    doc = nlp(text)

    text = ' '.join([token.lemma_ for token in doc])

    text = text.replace(" ,",",")

    text = text.replace(" .",".")

    return text


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):

    for contractions, base in CONTRACTION_MAP.items():
        text = text.replace(contractions, base)

    return text


def remove_accented_chars(text):

    nfkd_form = unicodedata.normalize('NFKD', text)
    text = u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

    return text


def remove_special_chars(text, remove_digits:bool=False):

    if remove_digits:
        text = re.sub("[^a-zA-Z ]", '', text)
    else:
        text = re.sub("[^a-zA-Z0-9 ]", '', text)
    

    return text


def remove_stopwords(text, is_lower_case=True, stopwords=stopword_list):

    if is_lower_case:
        text = text.lower()

    word_list = tokenizer.tokenize(text)

    clean_words = []
    for w in word_list:
        if w not in stopwords:
            clean_words.append(w)  
    
    return ' '.join(clean_words)


def remove_extra_new_lines(text):

     return re.sub('\n+', ' ', text)


def remove_extra_whitespace(text):
    
    return re.sub(' +', ' ', text)
    

def normalize_corpus(
    corpus,
    html_stripping=True,
    contraction_expansion=True,
    accented_char_removal=True,
    text_lower_case=True,
    text_stemming=False,
    text_lemmatization=True,
    special_char_removal=True,
    remove_digits=False,
    stopword_removal=True,
    stopwords=stopword_list
):
    
    normalized_corpus = []

    # Normalize each doc in the corpus
    for doc in corpus:
        # Remove HTML
        if html_stripping:
            doc = remove_html_tags(doc)
            
        # Remove extra newlines
        doc = remove_extra_new_lines(doc)
        
        # Remove accented chars
        if accented_char_removal:
            doc = remove_accented_chars(doc)
            
        # Expand contractions    
        if contraction_expansion:
            doc = expand_contractions(doc)
            
        # Lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)
            
        # Stemming text
        if text_stemming and not text_lemmatization:
            doc = stem_text(doc)
            
        # Remove special chars and\or digits    
        if special_char_removal:
            doc = remove_special_chars(
                doc,
                remove_digits=remove_digits
            )  

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)

         # Lowercase the text    
        if text_lower_case:
            doc = doc.lower()

        # Remove stopwords
        if stopword_removal:
            doc = remove_stopwords(
                doc,
                is_lower_case=text_lower_case,
                stopwords=stopwords
            )

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)
        doc = doc.strip()
            
        normalized_corpus.append(doc)
        
    return normalized_corpus

stop_words = nltk.corpus.stopwords.words('english')

def normalize_review(review: str) -> str:
    return normalize_corpus([review], stop_words)[0]
