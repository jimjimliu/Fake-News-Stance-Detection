import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import pandas as pd
from tqdm import tqdm


class Preprocess(object):

    def __init__(self, dataframe):
        self.df = dataframe.copy(deep=True)
        tqdm(desc=">>> Executing Preprocess.py").pandas()

    def preprocess(self, columns):
        '''

        :param columns: Single string or list of any length of strings; corresponding to column indexes of dataframe;
        :return: (DataFrame) cleaned Dataframe with order and structure untouched
        '''

        if(isinstance(columns, str)):
            self.df.loc[:][columns] = self.df.progress_apply(self.__filter, axis=1, column=columns)

        elif(isinstance(columns, list) and len(columns)>0):
            for col in columns:
                self.df.loc[:][col] = self.df.progress_apply(self.__filter, axis=1, column=col)

        return self.df


    def __filter(self, row, column):

        line = row[column]
        tokens = word_tokenize(line)
        # convert to lower case
        tokens = [w.lower() for w in tokens]
        # remove punctunation from each word
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens ]
        # remove remaining tokens are not alphabetic
        words = [word for word in stripped if word.isalpha()]
        # filter out stop words
        STOP_WORDS = set(stopwords.words('english'));
        words = [w for w in words if not w in STOP_WORDS]
        data = " ".join(words)

        return data


