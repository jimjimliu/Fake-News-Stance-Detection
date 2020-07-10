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


LABEL = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}

class FeatureExtract(object):

    def __init__(self, data=None, over_sampling=False):
        '''

        :param data: (np.array) has ['headline', 'body id', 'stance'] as rows
        :param over_sampling: (Boolean)
            If True, resampling data set using oversampling.
            Defualt=False
        '''
        self.__data = data
        self.__over_sam = over_sampling
        self.__embedding = os.path.join(os.getcwd(), 'data', "glove.6B.50d.txt")
        # build invese dict for data
        self.__inverse_doc = self.__inverse_doc()
        # build tf vector for head+body
        self.__tf_vector = self.data_to_tf(data)
        # build tf idf word corpus for entire news data set
        self.__tfidf_lookup = self.tfidf_lookup(np.r_[self.__data[:, 1], self.__data[:, 0]])
        self.__glove_vectors = self.__load_word_embedding()
        self.__x_train, self.__y_train = self.__extract_features()

    def __extract_features(self):

        ftrs = [self.cosine_similarity, self.kl_divergence, self.ngram_overlap]

        x_train = []
        for doc in tqdm(self.__data, desc="[Extracting features: ]"):
            vec = np.array([0.0] * len(ftrs))
            for i in range(len(ftrs)):
                vec[i] = ftrs[i](doc)
            # # get tf vector of head+body
            # index = self.__inverse_doc[doc.tobytes()]
            # vec = np.concatenate((vec, self.__tf_vector[index]))
            x_train.append(vec)
        x_train = np.array(x_train)
        y_train = np.array([LABEL[i] for i in self.__data[:, 2]])

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
        build a tf-idf lookup table for doc.

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
        f_glove = open(self.__embedding, "rb")  # download from https://nlp.stanford.edu/projects/glove/
        glove_vectors = {}
        for line in tqdm(f_glove, desc="[Loading in word embeddings: ]"):
            glove_vectors[str(line.split()[0]).split("'")[1]] = np.array(list(map(float, line.split()[1:])))

        return glove_vectors

    def doc_to_glove(self, doc):
        # Convert a document to GloVe vectors, by computing tf-idf of each word * GLoVe of word / total tf-idf for document

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
        headline_vector = self.doc_to_glove(doc[0])
        body_vector = self.doc_to_glove(doc[1])
        cos = cosine_similarity([headline_vector], [body_vector])[0][0]
        return cos

    def divergence(self, lm1, lm2):
        # Compute the KL-Divergence of language model (LM) representations of the headline and the body
        sigma = 0.0
        for i in range(lm1.shape[0]):  # assume lm1 and lm2 has same shape
            sigma += lm1[i] * np.log(lm1[i] / lm2[i])
        return sigma

    def kl_divergence(self, doc, eps=0.1):
        # Convert headline and body to 1-gram representations
        tf_headline = self.doc_to_tf(doc[0])
        tf_body = self.doc_to_tf(doc[1])

        # Convert dictionary tf representations to vectors (make sure columns match to the same word)
        words = set(tf_headline.keys()).union(set(tf_body.keys()))
        vec_headline, vec_body = np.zeros(len(words)), np.zeros(len(words))
        i = 0
        for word in words:
            vec_headline[i] += tf_headline[word]
            vec_body[i] = tf_body[word]
            i += 1

        # Compute a simple 1-gram language model of headline and body
        lm_headline = vec_headline + eps
        lm_headline /= np.sum(lm_headline)
        lm_body = vec_body + eps
        lm_body /= np.sum(lm_body)

        # Return KL-divergence of both language models
        return self.divergence(lm_headline, lm_body)

    def ngram_overlap(self, doc):
        # Returns how many times n-grams (up to 3-gram) that occur in the article's headline occur on the article's body.
        tf_headline = self.doc_to_tf(doc[0], ngram=3)
        tf_body = self.doc_to_tf(doc[1], ngram=3)
        matches = 0.0
        for words in tf_headline.keys():
            if words in tf_body:
                matches += tf_body[words]
        return np.power((matches / len(self.tokenise(doc[1]))), 1 / np.e)  # normalise for document length

    def get_X(self):
        return self.__x_train

    def get_y(self):
        return self.__y_train

    def get_idf_lookup(self):
        return self.__idf_lookup

    def get_tfidf_lookup(self):
        return self.__tfidf_lookup

if __name__ == '__main__':
    arr = ["Rare sighting of Anna Wintour without her trademark sunglasses as she leaves rat-infested Vogue office",
           "Homeland Security Secretary Jeh Johnson is pushing back against a claim made by Rep."]
    arr = np.array(arr)
    fe = FeatureExtract()
    print(fe.tokenise(arr[0]))
    print(word_tokenize(arr[0]))
