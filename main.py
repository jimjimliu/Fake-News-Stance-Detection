import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import re
from collections import defaultdict
from scorer import score_submission, print_confusion_matrix, score_defaults, SCORE_REPORT
from nltk import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import os
from imblearn.over_sampling import SMOTE
from DataSet import DataSet
from Feature_Extract import FeatureExtract
from utils import utils, SUBMISSION_PATH, MODEL_PATH, DATA_PATH
from sklearn.svm import SVC
from scorer import score_submission, print_confusion_matrix, score_defaults, SCORE_REPORT, load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from Preprocess import Preprocess
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import initializers
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers import Embedding, LSTM, GRU, GlobalMaxPool1D
from keras.utils import np_utils
from tensorflow.keras import regularizers
from sklearn.metrics import accuracy_score
from keras.utils import np_utils
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
from matplotlib import pyplot as plt


# CONFIGURATION PARAMETER
PREPROCESS = False
OVER_SAMPLING = False
TF_MAX_FEATURE = 5000

class main():

    def __init__(self, path=None):
        self.path = path
        self.__classifiers = self.classifiers()
        self.__model_score = {}
        # store predictions of one conventional classifier like SVM in self.__classifiers list
        # the order of the predictions is the same order of testing set
        self.half_pred = []


    def classify(self):
        '''

        :return:
        '''

        "prepare data"
        utils.print_info("1, prepareing data")
        ds = DataSet(preprocess=PREPROCESS)
        train_all = ds.get_train()
        val_all = ds.get_validation()
        test_all = ds.get_test()

        "extract features from training set"
        utils.print_info("2, extracting training features")
        # using NN model to train, 2 features with labels [agree, disagree, discuss]
        fe = FeatureExtract(train_all, over_sampling=OVER_SAMPLING, set='train', class_format='trinary')
        X_train_tri, y_train_tri = fe.get_X(), fe.get_y()
        # using SVM to train, 3 features, with label [related, unrelated]
        fe_bi = FeatureExtract(train_all, over_sampling=OVER_SAMPLING, set='train', class_format='binary')
        X_train_bi, y_train_bi = fe_bi.get_X(), fe_bi.get_y()

        "extract features from testing set"
        utils.print_info("3, extracting testing features")
        # using NN model to test, 2 features
        fe_test = FeatureExtract(test_all, set='test', class_format='trinary')
        X_test, y_test = fe_test.get_X(), fe_test.get_y()
        # using SVM to predict
        fe_test_bi = FeatureExtract(test_all, set='test', class_format='binary')
        X_test_bi, y_test_bi = fe_test_bi.get_X(), fe_test_bi.get_y()

        "fit model"
        utils.print_info("4, fitting/testing model")
        self.clf_comparison(X_train_bi, y_train_bi, X_test_bi, y_test_bi)

        "testing"
        utils.print_info("5, neural network model")
        self.NN_model(X_train_tri,y_train_tri,X_test,y_test)

    def classifiers(self):
        return [
            (SVC(C=50.0, cache_size=200, class_weight=None, coef0=0.0,
                 decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
                 max_iter=-1, probability=True, random_state=None, shrinking=True,
                 tol=0.001, verbose=False), 'SVM', 'SVM'),

            # (GradientBoostingClassifier(random_state=0), 'GBDT', 'GBDT'),
            #
            # (KNeighborsClassifier(1), "K NN ", 'KNN'),
            #
            # (QuadraticDiscriminantAnalysis(), 'Qudratic Discriminant Analysis', 'QD'),
            #
            # (RandomForestClassifier(max_depth=50, n_estimators=10, max_features=1), 'Random Forest Classifier', 'RF'),
            #
            # (AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=0.01,
            #                     algorithm='SAMME.R', random_state=None), 'Adaboost Classifier', 'Ada'),
            # (SGDClassifier(), 'SGD Classifier', 'SGD'),
            #
            # (DecisionTreeClassifier(max_depth=5), 'Decision Tree Classifier', 'DT'),
            #
            # (LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, n_components=None,
            #                             store_covariance=False, tol=0.00001), 'Linear Discriminant Analysis', 'LDA'),
            #
            # (GaussianNB(), 'Gaussian Naive Bayes ', 'GNB')
        ]

    def clf_comparison(self, X, y, X_test, y_test):
        '''
        Frist step of 2 step classification.
        using conventional classifier to train data set with only labels [related, unrelated].
        By using SVM, the accuracy of binary classification can be above 95%.
        store the predition result in global variable for future use.

        :param X: x train
        :param y: y train
        :param X_test:
        :param y_test:
        :return:
        '''

        competition_test = utils.read('competition_test_stances.csv')
        gold_labels = load_dataset(os.path.join(DATA_PATH, 'competition_test_stances.csv'))

        for model, name, initial in self.__classifiers:
            print("Fitting...")
            clf = model.fit(X, y)
            print("Predicting...")
            predictions = clf.predict(X_test)
            # store GDBT's prediction on agree and disagree
            self.half_pred = predictions
            print(name, " Testing score: ", clf.score(X_test, y_test))
            # save model
            utils.save_pkl_model(clf, initial)

    def NN_model(self, X, y, X_test, y_test):
        '''
        second step of 2 step classification.
        using neural network to classify 3 labels [agree, disagree, discuss].

        :param X:
        :param y:
        :param X_test:
        :param y_test:
        :return:
        '''
        y = np_utils.to_categorical(y, 3)
        y_true = y_test
        # y_test = np_utils.to_categorical(y_test, 3)

        init = initializers.glorot_uniform(seed=1)
        model = Sequential()
        model.add(Dense(units=20, input_dim=2,kernel_initializer=init,  activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(units=10, kernel_initializer=init, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(units=3, kernel_initializer=init, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
        model.summary()
        model.fit(X, y, batch_size=5, epochs=120, verbose=2)
        # loss, accuracy = model.evaluate(X_test, y_test)
        # print('Accuracy: %f' % (accuracy * 100))
        # exit()
        predictions = np.argmax(model.predict(X_test), axis=-1)

        "combine clf and NN clf's result together"
        predictions = self.__combine(self.half_pred, predictions)

        print(accuracy_score(y_true, predictions))

        "save the model"
        model.save(os.path.join(os.getcwd(), 'model','FNN.h5'))

        "write out submission file"
        utils.print_info("5, Writing submission results")
        competition_test = utils.read('competition_test_stances.csv')
        print(len(competition_test), len(predictions))
        utils.write_out(competition_test, predictions, 'answer')

        "print confusion matrix and score"
        gold_labels = load_dataset(os.path.join(DATA_PATH, 'competition_test_stances.csv'))
        test_labels = load_dataset(os.path.join(SUBMISSION_PATH, 'answer.csv'))
        test_score, cm = score_submission(gold_labels, test_labels)
        null_score, max_score = score_defaults(gold_labels)
        print_confusion_matrix(cm)
        print(SCORE_REPORT.format(max_score, null_score, test_score))
        self.__model_score['FNN'] = [test_score, print_confusion_matrix(cm)]

    def __combine(self, clf_pred, nn_pred):
        '''
        clf_pred contains prediction of 2 labels [0,1] which 0 is related, 1 is unrelated.
        nn_pred contains prediction of 3 labels [0,1,2] which 0 is agree, 1 is disagree, 2 is discuss.
        Combine two predictions together, keep classified label 'unrelated' untouched, and for 'related',
        using predictions in nn_pred to swap.

        :param clf_pred: (ndarray)
        :param nn_pred: (ndarray)
        :return: (ndarray)
        '''
        predictions = []
        for i in range(len(clf_pred)):
            # 如果是1,代表unrelated, 变成3
            if clf_pred[i] == 1:
                predictions.append(3)
            else:
                predictions.append(nn_pred[i])

        return np.array(predictions)


if __name__ == '__main__':
    "before running, download word_embedding file at https://nlp.stanford.edu/projects/glove/"
    "https://www.kaggle.com/pkugoodspeed/nlpword2vecembeddingspretrained?select=glove.6B.50d.txt"
    "put the glove.6B.50d.txt into data folder and you are good to go"
    main().classify()






