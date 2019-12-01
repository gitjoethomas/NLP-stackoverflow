from bs4 import BeautifulSoup
import nltk
import pandas as pd
import numpy as np


def find_number_answers(answers_df):
    """for each question, works out how many answers were given"""
    
    number_answers = answers_df['parentid'].value_counts() # get counts
    number_answers = pd.DataFrame(data = number_answers) # convert to df
    
    number_answers.index.rename('parentid', inplace = True) # renaming
    number_answers.columns = ['number_answers'] # renaming

    return number_answers

def is_questioner_reply(answers_df, questions_df):
    '''
    returns a copy of the answers_df with an extra column (is_questioner flag) to indicate whether the post is by the user
    who originally asked the question
    '''
    
    # link answers back to their question on question_id == answer's parent_id
    combined = answers_df.merge(questions_df, left_on = 'parentid', right_index = True, suffixes = ['_answers', '_questions'])
    
    # the user who wrote the answer must be the same as the user who wrote the question
    filtered = combined[combined['owneruserid_answers'] == combined['owneruserid_questions']].copy()
    
    # is_questioner is a 1 if the answer is written by the orinal questioner, and a 0 if not
    filtered['is_questioner'] = 1
    filtered = filtered[['is_questioner']]
    answers_df = answers_df.merge(filtered, left_index = True, right_index = True, how = 'left')
    answers_df = answers_df.fillna(0)
    
    return answers_df

def is_top_answer(answers_df):
    """creates column is_top_answer - a 1/0 flag for whether the answer has the highest score for the question"""
    func_df = answers.copy()
    
    # for each question, answers are ranked by score, highest score first
    func_df['rank'] = func_df.groupby('parentid')['score'].rank(method = 'max',ascending = False).copy()
    
    func_df['is_top_answer'] = func_df['rank'].copy()
    func_df.loc[func_df['rank'] != 1, 'is_top_answer'] = 0 # this needs .loc to avoid a genuine settingwithcopy() problem.
    
    return func_df

def remove_html(cell):
    """takes out html tags and NLTK stopwords from a block of text. call with .apply()"""
    
    # BeautifulSoup will parse out html tags
    clean = BeautifulSoup(cell, features="html5lib").get_text() 
    no_delimiters = clean.replace("\n", "")
        
    return no_delimiters


def remove_stopwords(cell):
    """takes out html tags and NLTK stopwords from a block of text. call with .apply()"""
    
    # drop all the words which are english stopwords.
    
    try:
        no_stopwords = "".join(w+" " for w in nltk.word_tokenize(cell) 
                           if not w.lower() in nltk.corpus.stopwords.words('english'))
    except TypeError:
        return np.nan
    return no_stopwords