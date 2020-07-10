import pandas as pd
import csv
import re
import os
import numpy as np
from utils import utils
import random
from sklearn.model_selection import train_test_split
from Preprocess import Preprocess

LABELS = ['agree', 'disagree', 'discuss', 'unrelated']

class DataSet():

    def __init__(self, preprocess=False):
        '''
        Reading in article data from ../Data folder and generate dataframes contain news data;

        :param preprocess: (Boolean) True, to clean data set. Deault=False
        '''
        self.__preprocess = preprocess  # if yes, clean data set
        self.__data_folder = os.path.join(os.getcwd(), 'data')
        self.__train_bodies_csv = os.path.join(self.__data_folder, 'train_bodies.csv')
        self.__train_stance_csv = os.path.join(self.__data_folder, 'train_stances.csv')
        self.__test_bodies_csv = os.path.join(self.__data_folder, 'test_bodies.csv')
        self.__test_stances_csv = os.path.join(self.__data_folder, 'competition_test_stances.csv')
        self.__train_all, self.__val_all, self.__test_all = self.__reader()

    def __reader(self):
        '''

        :return:
        '''

        "read in training sets"
        train_bodies_df = pd.read_csv(r'' + self.__train_bodies_csv, delimiter=',', header=0, sep='\t',
                                   names=['body_id', 'article'])
        train_stances_df = pd.read_csv(r'' + self.__train_stance_csv, delimiter=',', header=0, sep='\t',
                                    names=['headline', 'body_id', 'stance'])
        train_stances_df['target'] = -1
        # assign target number to each type of target name
        for i in range(len(LABELS)):
            train_stances_df.loc[train_stances_df['stance'] == LABELS[i], 'target'] = i

        "read in testing sets"
        test_bodies_df = pd.read_csv(r'' + self.__test_bodies_csv, delimiter=',', header=0, sep='\t',
                                      names=['body_id', 'article'])
        test_stances_df = pd.read_csv(r'' + self.__test_stances_csv, delimiter=',', header=0, sep='\t',
                                       names=['headline', 'body_id', 'stance'])
        test_stances_df['target'] = -1
        # assign target number to each type of target name
        for i in range(len(LABELS)):
            test_stances_df.loc[test_stances_df['stance'] == LABELS[i], 'target'] = i

        # left join tow dataframes
        train_df = pd.merge(train_stances_df, train_bodies_df, on='body_id', how='left')
        test_df = pd.merge(test_stances_df, test_bodies_df, on='body_id', how='left')
        # clean two data sets
        if(self.__preprocess):
            print('Cleaning training set...\n')
            train_df = Preprocess(train_df).preprocess(['headline', 'article'])
            print('\nCleaning testing set...\n')
            test_df = Preprocess(test_df).preprocess(['headline', 'article'])
        # re-order column index, and drop some columns
        train_df = train_df[['headline','article','stance']]
        test_df = test_df[['headline', 'article', 'stance']]
        # use target labels to uniformly split data set
        train_all, val_all = train_test_split(train_df, train_size=0.9, random_state=0, stratify=train_df['stance'])

        all_unrelated = train_df[train_df['stance'] == 'unrelated']
        all_discuss = train_df[train_df['stance'] == 'discuss']
        all_agree = train_df[train_df['stance'] == 'agree']
        all_disagree = train_df[train_df['stance'] == 'disagree']
        train_unrelated = train_all[train_all['stance'] == 'unrelated']
        train_discuss = train_all[train_all['stance'] == 'discuss']
        train_agree = train_all[train_all['stance'] == 'agree']
        train_disagree = train_all[train_all['stance'] == 'disagree']
        val_unrelated = val_all[val_all['stance'] == 'unrelated']
        val_discuss = val_all[val_all['stance'] == 'discuss']
        val_agree = val_all[val_all['stance'] == 'agree']
        val_disagree = val_all[val_all['stance'] == 'disagree']

        print('\n\tUnrltd\tDiscuss\t Agree\tDisagree')
        print('All\t', len(all_unrelated), '\t', len(all_discuss), '\t', len(all_agree), '\t', len(all_disagree))
        print('Train\t', len(train_unrelated), '\t', len(train_discuss), '\t', len(train_agree), '\t',
              len(train_disagree))
        print('Valid.\t', len(val_unrelated), '\t', len(val_discuss), '\t', len(val_agree), '\t', len(val_disagree))

        train_all = np.array(train_all)
        val_all = np.array(val_all)
        test_all = np.array(test_df)

        return train_all, val_all, test_all

    def get_train(self):
        return self.__train_all

    def get_validation(self):
        return self.__val_all

    def get_test(self):
        return self.__test_all

