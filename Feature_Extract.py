from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from DataSet import DataSet
from scipy.sparse import hstack,vstack,csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import numpy as np
import pandas as pd
import os
import pickle
from utils import utils
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
import random

class FeatureExtract(object):

    def __init__(self, data=None, over_sampling=False, class_format=None, set='train'):
        '''

        :param data: (np.array) has ['headline', 'body id', 'stance'] as rows
        :param over_sampling: (Boolean)
            If True, resampling data set using oversampling.
            Defualt=False
        :param separate: (String) has the value of [binary, trinary].
            if binary, means to convert the 4 labels of input data set to 2 labels.
            change[agree, disagree, discuss] to related, and keep label 'unrelated' untouch
        :param set: (String) has values [train, test]
        '''
        self.__class_format = class_format
        self.__set = set
        self.label = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}
        self.__data, self.label = self.change_data_format(data, class_format, set)
        self.__over_sam = over_sampling
        self.__embedding = os.path.join(os.getcwd(), 'data', "glove.6B.50d.txt")
        # build inverse dict for data
        self.__inverse_doc = self.__inverse_doc()
        # build tf vector for head+body
        self.__tf_vector = self.data_to_tf(data)
        # build tf idf word corpus for entire news data set
        self.__tfidf_lookup = self.tfidf_lookup(np.r_[self.__data[:, 1], self.__data[:, 0]])
        self.__glove_vectors = self.__load_word_embedding()
        self.__x_train, self.__y_train = self.__extract_features()

    def change_data_format(self, data, class_format, set):
        '''
        input data set has 4 labels, which is [agree, disagree, discuss, unrelated].
        Given class_format, change 4 labels to 2 or 3.
        data set with 2 labels is for binary classification, to be fed into SVM classifier(or you can
        change the classifier in main.py).
        Data with 3 labels is to be fed into neural network model.

        :param data: (ndarray) [[head,body,stance]], has labels [agree, disagree, discuss, unrelated].
        :param class_format: (String) has values [binary, trinary]
            indicates how many labels left after changing.
            if binary, change [agree, disagree, discuss] to related. keep unrelated untouch.
            if trinary, delete rows with label unrelated. keep only [agree, disagree, discuss] for classification.
        :param set: (String) has values [train, test].
            if train, the input data set needs to be changed to binary or trinary format.
            if test, the input data set remains the same.
            i.e. when trainning, the binary training set has 25413 rows with 2 labels. Trinary set has less rows with
            3 labels. When testing, the data set remains 25413 rows and row order unchanged.
        :return: (ndarrray) new data set with 2 or 3 labels.
        '''

        if set == 'train':
            if class_format == 'binary':
                # data set to feed conventional classifier
                new_data = self.to_binary_labels(data)
                label = {'related': 0, 'unrelated': 1}
            if class_format == 'trinary':
                # data set to feed neural network
                label = {'agree': 0, 'disagree': 1, 'discuss': 2}
                new_data = self.to_trinary_labels(data)
        else:
            new_data = data
            label = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}

        return new_data, label

    def to_binary_labels(self, data):
        '''
        convert the original data set, change labels [agree, disagree, discuss] to [related]
        for conventional classifier to train. keep unrelated label untouched

        :param data: (ndarray) [[head,body,stance]], has labels [agree, disagree, discuss, unrelated].
        :return: (ndarray) [[head,body,stance]],
        '''
        new_data = pd.DataFrame(data, columns=['head','body','stance'])
        new_data.loc[new_data['stance']=='discuss', 'stance'] = 'related'
        new_data.loc[new_data['stance'] == 'agree', 'stance'] = 'related'
        new_data.loc[new_data['stance'] == 'disagree', 'stance'] = 'related'
        # shuffle set
        new_data = new_data.sample(frac=1)
        # keep the order of index after shuffle
        index = new_data[:].index.tolist()
        new_data = np.array(new_data)
        return new_data

    def to_trinary_labels(self, data):
        '''
        select a subset from original data set. choose rows that have label
        of [agree, disagree, discuss] for neural network to train.

        :param data: (ndarray) [[head,body,stance]], has labels [agree, disagree, discuss, unrelated].
        :return: (ndarray) [[head,body,stance]],
        '''
        new_data = pd.DataFrame(data, columns=['head','body','stance'])
        new_data = pd.concat([new_data[new_data['stance']=='agree'], new_data[new_data['stance']=='disagree'], new_data[new_data['stance']=='discuss']])
        # shuffle
        new_data = new_data.sample(frac=1)
        # keep the order of rows after shuffle
        index = new_data[:].index.tolist()
        new_data = np.array(new_data)
        return new_data

    def __extract_features(self):
        '''
        using self.__data's [head, body] to extract features.

        :return: (ndarray), (ndarray)
        '''

        # if trinary, means using neural network to classify [agree, disagree, discuss] using 2 features
        if self.__class_format == 'trinary':
            ftrs = [self.cosine_similarity, self.kl_divergence]
        else:
            ftrs = [self.cosine_similarity, self.kl_divergence, self.ngram_overlap]

        x_train = []
        for doc in tqdm(self.__data, desc="[Extracting features: ]"):
            vec = np.array([0.0] * len(ftrs))
            for i in range(len(ftrs)):
                vec[i] = ftrs[i](doc)
            x_train.append(vec)
        x_train = np.array(x_train)
        y_train = np.array([self.label[i] for i in self.__data[:, 2]])

        if(self.__over_sam):
            print("Over-sampling...")
            sm = SMOTE(random_state=0)
            x_train, y_train = sm.fit_resample(x_train, y_train)

        return x_train, y_train

    def __inverse_doc(self):
        '''
        build a inverse lookup dict for [[head, body, stance]].
        each item [head,body,stance] is the key. row index is the value

        :return: (Dict)
        '''

        # build inverse dict for data, key is [head, body, stance], value is row index
        dict = {}
        for i in tqdm(range(len(self.__data)), desc="[Building inverse doc: ]"):
            dict[self.__data[i].tobytes()] = i
        return  dict

    def data_to_tf(self, doc):
        '''
        Build tf vectors for doc. Combine head+body together and using vector of size 5000 to represent it.

        :param doc: (ndarray) [[head, body, stance]].
        :return: Tf-idf-weighted document-term matrix
        '''

        # row[0] is head, row[1] is body
        data = [row[0]+" "+row[1] for row in tqdm(doc, desc="[Building tf vectors: ]")]
        tf_vectorizer = TfidfVectorizer(use_idf=False, token_pattern=r"(?u)\b\w+\b", max_features=5000)
        tf = tf_vectorizer.fit_transform(data).toarray()

        return tf

    def tokenise(self, text):
        pattern = re.compile("[^a-zA-Z0-9 ]+")
        stop_words = set(stopwords.words('english'))
        text = pattern.sub('', text.replace('\n', ' ').replace('-', ' ').lower())
        text = [word for word in word_tokenize(text) if word not in stop_words]
        return text

    def doc_to_tf(self, text, ngram=1):
        words = self.tokenise(text)
        ret = defaultdict(float)
        for i in range(len(words)):
            for j in range(1, ngram + 1):
                if i - j < 0:
                    break
                word = [words[i - k] for k in range(j)]
                ret[word[0] if ngram == 1 else tuple(word)] += 1.0
        return ret

    def tfidf_lookup(self, doc):
        '''
        build a tf-idf lookup table for distinct words in doc.

        :param doc: iterable
            An iterable which yields either str, unicode or file objects.
        :return: (Dict)
            tf-idf lookup dictionary{word, tf-idf}
        '''
        print("Building tf-idf vectors...")
        data = [' '.join(self.tokenise(sentence)) for sentence in doc]

        # build tf-idf lookup table, token_pattern keeps length of 1 words
        tfidf_vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, token_pattern=r"(?u)\b\w+\b")
        tfidf = tfidf_vectorizer.fit_transform(data)
        coo_matrix = tfidf.tocoo()
        tuples = zip(coo_matrix.col, coo_matrix.data)
        sorted_items = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
        feature_names =tfidf_vectorizer.get_feature_names()

        word_idf = {}
        # word index and corresponding tf-idf score
        for idx, score in tqdm(sorted_items, desc="[Building tf-idf lookup table: ]"):
            word_idf[feature_names[idx]] = round(score, 3)
        # return a sorted list of tuples with feature name and tf-idf score as its element(in descending order of tf-idf scores).
        return word_idf

    def __load_word_embedding(self):
        # Load GLoVe word vectors
        # download from https://nlp.stanford.edu/projects/glove/
        f_glove = open(self.__embedding, "rb")
        glove_vectors = {}
        for line in tqdm(f_glove, desc="[Loading in word embeddings: ]"):
            glove_vectors[str(line.split()[0]).split("'")[1]] = np.array(list(map(float, line.split()[1:])))

        return glove_vectors

    def doc_to_glove(self, doc):
        '''
        Convert a document to GloVe vectors, by computing tf-idf of each word * GLoVe of word / total tf-idf for document.
        project each word token in doc onto GLoVe word embedding vectors using tf-idf of the word.

        :param doc: (String) string of words.
        :return: (ndarray)
        '''

        words_lst = self.tokenise(doc)
        # initialize vector of length as length of word embedding
        doc_vector = np.zeros(self.__glove_vectors['glove'].shape[0])
        if np.sum(list(self.__tfidf_lookup.values())) == 0.0:  # edge case: document is empty
            return doc_vector

        for word in words_lst:
            if word in self.__glove_vectors and word in self.__tfidf_lookup:
                doc_vector += self.__glove_vectors[word] * self.__tfidf_lookup[word]
        doc_vector /= np.sum(list(self.__tfidf_lookup.values()))
        return doc_vector

    def cosine_similarity(self, doc):
        '''

        :param doc: (ndarray) [head,body,stance]
        :return: (Float)
        '''
        headline_vector = self.doc_to_glove(doc[0])
        body_vector = self.doc_to_glove(doc[1])
        cos = cosine_similarity([headline_vector], [body_vector])[0][0]
        return cos

    def divergence(self, lm1, lm2):
        sigma = 0.0
        for i in range(lm1.shape[0]):
            sigma += lm1[i] * np.log(lm1[i] / lm2[i])
        return sigma

    def kl_divergence(self, doc, eps=0.1):

        tf_headline = self.doc_to_tf(doc[0])
        tf_body = self.doc_to_tf(doc[1])

        words = set(tf_headline.keys()).union(set(tf_body.keys()))
        vec_headline, vec_body = np.zeros(len(words)), np.zeros(len(words))
        i = 0
        for word in words:
            vec_headline[i] += tf_headline[word]
            vec_body[i] = tf_body[word]
            i += 1

        lm_headline = vec_headline + eps
        lm_headline /= np.sum(lm_headline)
        lm_body = vec_body + eps
        lm_body /= np.sum(lm_body)

        return self.divergence(lm_headline, lm_body)

    def ngram_overlap(self, doc):
        tf_headline = self.doc_to_tf(doc[0], ngram=3)
        tf_body = self.doc_to_tf(doc[1], ngram=3)
        matches = 0.0
        for words in tf_headline.keys():
            if words in tf_body:
                matches += tf_body[words]
        return np.power((matches / len(self.tokenise(doc[1]))), 1 / np.e)

    def get_X(self):
        return self.__x_train

    def get_y(self):
        return self.__y_train

    def get_tfidf_lookup(self):
        return self.__tfidf_lookup


