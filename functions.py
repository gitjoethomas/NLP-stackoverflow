from bs4 import BeautifulSoup
import nltk


def remove_html(cell):
    """takes out html tags and NLTK stopwords from a block of text. call with .apply()"""
    
    # BeautifulSoup will parse out html tags
    clean = BeautifulSoup(cell, "html5").get_text() 
    no_delimiters = clean.replace("\n", "")
        
    return no_delimiters


def remove_stopwords(cell):
    """takes out html tags and NLTK stopwords from a block of text. call with .apply()"""
    
    # drop all the words which are english stopwords.
    no_stopwords = "".join(w+" " for w in nltk.word_tokenize(cell) 
                           if not w.lower() in nltk.corpus.stopwords.words('english'))
        
    return no_stopwords