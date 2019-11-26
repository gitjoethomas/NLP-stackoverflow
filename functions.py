from bs4 import BeautifulSoup
import nltk
import pandas as pd


def find_number_answers(answers_df):
    """for each question, works out how many answers were given"""
    
    number_answers = answers_df['parentid'].value_counts() # get counts
    number_answers = pd.DataFrame(data = number_answers) # convert to df
    
    number_answers.index.rename('parentid', inplace = True) # renaming
    number_answers.columns = ['number_answers'] # renaming

    return number_answers


def remove_html(cell):
    """takes out html tags and NLTK stopwords from a block of text. call with .apply()"""
    
    # BeautifulSoup will parse out html tags
    clean = BeautifulSoup(cell, "html5").get_text() 
    no_delimiters = clean.replace("\n", "")
        
    return no_delimiters


def remove_stopwords(cell):
    """takes out html tags and NLTK stopwords from a block of text. call with .apply()"""
    
    # drop all the words which are english stopwords.
    
    try:
        no_stopwords = "".join(w+" " for w in nltk.word_tokenize(cell) 
                           if not w.lower() in nltk.corpus.stopwords.words('english'))
    except Exception:
        print("cell is: "+str(cell))
        raise Exception
    return no_stopwords