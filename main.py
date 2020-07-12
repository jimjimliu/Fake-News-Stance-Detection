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


# CONFIGURATION PARAMETER
PREPROCESS = True
OVER_SAMPLING = False
TF_MAX_FEATURE = 5000

class main():

    def __init__(self, path=None):
        self.path = path
        self.__classifiers = self.classifiers()
        self.__model_score = {}

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
        fe = FeatureExtract(train_all, over_sampling=OVER_SAMPLING)
        X_train, y_train = fe.get_X(), fe.get_y()

        "extract features from testing set"
        utils.print_info("3, extracting testing features")
        fe_test = FeatureExtract(test_all)
        X_test, y_test = fe_test.get_X(), fe_test.get_y()

        "fit model"
        utils.print_info("4, fitting/testing model")
        self.clf_comparison(X_train, y_train, X_test, y_test)

        "testing"
        utils.print_info("5, neural network model")
        self.NN_model(X_train,y_train,X_test,y_test)

        "ploting"
        utils.print_info("6, plotting")
        self.show_plot()

    def classifiers(self):
        return [
            (SVC(C=50.0, cache_size=200, class_weight=None, coef0=0.0,
                 decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
                 max_iter=-1, probability=True, random_state=None, shrinking=True,
                 tol=0.001, verbose=False), 'SVM', 'SVM'),

            (GradientBoostingClassifier(random_state=0), 'GBDT', 'GBDT'),

            (KNeighborsClassifier(1), "K NN ", 'KNN'),

            (QuadraticDiscriminantAnalysis(), 'Qudratic Discriminant Analysis', 'QD'),

            (RandomForestClassifier(max_depth=50, n_estimators=10, max_features=1), 'Random Forest Classifier', 'RF'),

            (AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=0.01,
                                algorithm='SAMME.R', random_state=None), 'Adaboost Classifier', 'Ada'),
            (SGDClassifier(), 'SGD Classifier', 'SGD'),

            (DecisionTreeClassifier(max_depth=5), 'Decision Tree Classifier', 'DT'),

            (LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, n_components=None,
                                        store_covariance=False, tol=0.00001), 'Linear Discriminant Analysis', 'LDA'),

            (GaussianNB(), 'Gaussian Naive Bayes ', 'GNB')
        ]

    def clf_comparison(self, X, y, X_test, y_test):
        competition_test = utils.read('competition_test_stances.csv')
        gold_labels = load_dataset(os.path.join(DATA_PATH, 'competition_test_stances.csv'))

        for model, name, initial in self.__classifiers:
            print("Fitting...")
            clf = model.fit(X, y)
            print("Predicting...")
            predictions = clf.predict(X_test)
            print(name, " Testing score: ", clf.score(X_test, y_test))
            utils.write_out(competition_test, predictions, name)

            "print confusion matrix and score"
            test_labels = load_dataset(os.path.join(SUBMISSION_PATH, name+'.csv'))
            test_score, cm = score_submission(gold_labels, test_labels)
            null_score, max_score = score_defaults(gold_labels)
            print_confusion_matrix(cm)
            print(SCORE_REPORT.format(max_score, null_score, test_score))
            self.__model_score[initial] = [test_score, print_confusion_matrix(cm)]

    def NN_model(self, X, y, X_test, y_test):
        y = np_utils.to_categorical(y, 4)
        y_true = y_test
        y_test = np_utils.to_categorical(y_test, 4)

        init = initializers.glorot_uniform(seed=1)
        model = Sequential()
        model.add(Dense(units=20, input_dim=3,kernel_initializer=init,  activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(units=10, kernel_initializer=init, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(units=4, kernel_initializer=init, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
        model.summary()
        model.fit(X, y, batch_size=10, epochs=60, validation_data=(X_test, y_test), verbose=2)
        loss, accuracy = model.evaluate(X_test, y_test)
        print('Accuracy: %f' % (accuracy * 100))
        predictions = np.argmax(model.predict(X_test), axis=-1)
        print(accuracy_score(y_true, predictions))

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

    def show_plot(self):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        metrics = np.array(list(self.__model_score.values()))
        accu = metrics[:, 1] #accuracy
        scores = metrics[:, 0] #scores
        plt.plot(accu)
        for i, label in enumerate(list(self.__model_score.keys())):
            plt.text(i, accu[i], label)
        plt.show()

        plt.plot(scores)
        for i, label in enumerate(list(self.__model_score.keys())):
            plt.text(i, scores[i], label)
        plt.show()


if __name__ == '__main__':
    main().classify()






