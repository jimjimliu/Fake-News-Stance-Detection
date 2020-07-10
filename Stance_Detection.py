#!/usr/bin/env python
# coding: utf-8

# ## Loading the important Libraries

# In[1]:


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


# # Loading the contents

# In[2]:

"------------------------------------preparing data set--------------------------------------------------"
f_bodies = open(os.path.join(os.getcwd(), 'data', 'train_bodies.csv'), 'r', encoding='utf-8')
csv_bodies = csv.DictReader(f_bodies)
bodies = []
for row in csv_bodies:
    body_id = int(row['Body ID'])
    if (body_id + 1) > len(bodies):
        bodies += [None] * (body_id + 1 - len(bodies))
    bodies[body_id] = row['articleBody']
f_bodies.close()
body_inverse_index = {bodies[i]: i for i in range(len(bodies))}

all_unrelated, all_discuss, all_agree, all_disagree = [], [], [], []  # each article = (headline, body, stance)

f_stances = open(os.path.join(os.getcwd(), 'data', 'train_stances.csv'), 'r', encoding='utf-8')
csv_stances = csv.DictReader(f_stances)
for row in csv_stances:
    body = bodies[int(row['Body ID'])]
    if row['Stance'] == 'unrelated':
        all_unrelated.append((row['Headline'], body, row['Stance']))
    elif row['Stance'] == 'discuss':
        all_discuss.append((row['Headline'], body, row['Stance']))
    elif row['Stance'] == 'agree':
        all_agree.append((row['Headline'], body, row['Stance']))
    elif row['Stance'] == 'disagree':
        all_disagree.append((row['Headline'], body, row['Stance']))
f_stances.close()


# In[3]:


print('\tUnrltd\tDiscuss\t Agree\tDisagree')
print('All\t', len(all_unrelated), '\t', len(all_discuss), '\t', len(all_agree), '\t', len(all_disagree))
train_unrelated = all_unrelated[:len(all_unrelated) * 9 // 10]
train_discuss = all_discuss[:len(all_discuss) * 9 // 10]
train_agree = all_agree[:len(all_agree) * 9 // 10]
train_disagree = all_disagree[:len(all_disagree) * 9 // 10]

val_unrelated = all_unrelated[len(all_unrelated) * 9 // 10:]
val_discuss = all_discuss[len(all_discuss) * 9 // 10:]
val_agree = all_agree[len(all_agree) * 9 // 10:]
val_disagree = all_disagree[len(all_disagree) * 9 // 10:]

train_unrelated = all_unrelated[:len(all_unrelated) //100]
train_discuss = all_discuss[:len(all_discuss) //100]
train_agree = all_agree[:len(all_agree) //100]
train_disagree = all_disagree[:len(all_disagree) //100]


val_unrelated = all_unrelated[len(all_unrelated) * 9 // 10:]
val_discuss = all_discuss[len(all_discuss) * 9 // 10:]
val_agree = all_agree[len(all_agree) * 9 // 10:]
val_disagree = all_disagree[len(all_disagree) * 9 // 10:]

val_unrelated = val_unrelated[len(val_unrelated) * 9 // 10:]
val_discuss = val_discuss[len(val_discuss) * 9 // 10:]
val_agree = val_agree[len(val_agree) * 9 // 10:]
val_disagree = val_disagree[len(val_disagree) * 9 // 10:]



print('Train\t', len(train_unrelated), '\t', len(train_discuss), '\t', len(train_agree), '\t', len(train_disagree))
print('Valid.\t', len(val_unrelated), '\t', len(val_discuss), '\t', len(val_agree), '\t', len(val_disagree))

# # Uniform distribution of Data

# In[4]:
train_all = (train_unrelated + train_discuss + train_agree + train_disagree)

# each article = (headline, body, stance)
# random.Random(0).shuffle(train_all)
train_all = np.array(train_all)
# print(train_all[:1])
# print(type(train_all))
# print(train_all.shape)
# exit()

val_all = val_unrelated + val_discuss + val_agree + val_disagree
# random.Random(0).shuffle(val_all)
val_all = np.array(val_all)
# print(val_all.shape)
# exit()


# from DataSet import DataSet
# ds = DataSet()
# train_all = ds.get_train()
# val_all = ds.get_validation()

# Tokenise text
pattern = re.compile("[^a-zA-Z0-9 ]+")  # strip punctuation, symbols, etc.
stop_words = set(stopwords.words('english'))
def tokenise(text):
    text = pattern.sub('', text.replace('\n', ' ').replace('-', ' ').lower())
    text = [word for word in word_tokenize(text) if word not in stop_words]
    return text


# Compute term-frequency of words in documents
def doc_to_tf(text, ngram=1):
    words = tokenise(text)
    ret = defaultdict(float)
    for i in range(len(words)):
        for j in range(1, ngram+1):
            if i - j < 0:
                break
            word = [words[i-k] for k in range(j)]
            ret[word[0] if ngram == 1 else tuple(word)] += 1.0
    return ret


# Build corpus of article bodies and headlines in training dataset
corpus = np.r_[train_all[:, 1], train_all[:, 0]]  # 0 to 44973 are bodies, 44974 to 89943 are headlines
# print(type(corpus))
# print(corpus[:2])
# exit()
# Learn idf of every word in the corpus
df = defaultdict(float)
for doc in tqdm(corpus):
    words = tokenise(doc)
    seen = set()
    for word in words:
        if word not in seen:
            df[word] += 1.0
            seen.add(word)
num_docs = corpus.shape[0]
idf = defaultdict(float)
for word, val in tqdm(df.items()):
    idf[word] = np.log((1.0 + num_docs) / (1.0 + val)) + 1.0  # smoothed idf

# from Feature_Extract import FeatureExtract
# idf = FeatureExtract(train_all).build_corpus(train_all)
# print(list(idf.items())[:10])


# In[11]:


# Load GLoVe word vectors
f_glove = open("data/glove.6B.50d.txt", "rb")  # download from https://nlp.stanford.edu/projects/glove/
glove_vectors = {}
for line in tqdm(f_glove):
    glove_vectors[str(line.split()[0]).split("'")[1]] = np.array(list(map(float, line.split()[1:])))


# In[12]:

# Convert a document to GloVe vectors, by computing tf-idf of each word * GLoVe of word / total tf-idf for document
def doc_to_glove(doc):
    doc_tf = doc_to_tf(doc)
    doc_tf_idf = defaultdict(float)
    for word, tf in doc_tf.items():
        doc_tf_idf[word] = tf * idf[word]
        
    doc_vector = np.zeros(glove_vectors['glove'].shape[0])
    if np.sum(list(doc_tf_idf.values())) == 0.0:  # edge case: document is empty
        return doc_vector
    
    for word, tf_idf in doc_tf_idf.items():
        if word in glove_vectors:
            doc_vector += glove_vectors[word] * tf_idf
    doc_vector /= np.sum(list(doc_tf_idf.values()))
    return doc_vector

# Compute cosine similarity of GLoVe vectors for all headline-body pairs
def dot_product(vec1, vec2):
    sigma = 0.0
    for i in range(vec1.shape[0]):  # assume vec1 and vec2 has same shape
        sigma += vec1[i] * vec2[i]
    return sigma
    
def magnitude(vec):
    return np.sqrt(np.sum(np.square(vec)))
        
def cosine_similarity(doc):
    headline_vector = doc_to_glove(doc[0])
    body_vector = doc_to_glove(doc[1])
    
    if magnitude(headline_vector) == 0.0 or magnitude(body_vector) == 0.0:  # edge case: document is empty
        return 0.0
    
    return dot_product(headline_vector, body_vector) / (magnitude(headline_vector) * magnitude(body_vector))



# Compute the KL-Divergence of language model (LM) representations of the headline and the body
def divergence(lm1, lm2):
    sigma = 0.0
    for i in range(lm1.shape[0]):  # assume lm1 and lm2 has same shape
        sigma += lm1[i] * np.log(lm1[i] / lm2[i])
    return sigma

def kl_divergence(doc, eps=0.1):
    # Convert headline and body to 1-gram representations
    tf_headline = doc_to_tf(doc[0])
    tf_body = doc_to_tf(doc[1])
    
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
    return divergence(lm_headline, lm_body)


# Other feature 1
def ngram_overlap(doc):
    # Returns how many times n-grams (up to 3-gram) that occur in the article's headline occur on the article's body.
    tf_headline = doc_to_tf(doc[0], ngram=3)
    tf_body = doc_to_tf(doc[1], ngram=3)
    matches = 0.0
    for words in tf_headline.keys():
        if words in tf_body:
            matches += tf_body[words]
    return np.power((matches / len(tokenise(doc[1]))), 1 / np.e)  # normalise for document length

'---------------------------------------extract features------------------------------------------------'
# Define function to convert (headline, body) to feature vectors for each document
ftrs = [cosine_similarity, kl_divergence, ngram_overlap]
def to_feature_array(doc):
    vec = np.array([0.0] * len(ftrs))
    for i in range(len(ftrs)):
        vec[i] = ftrs[i](doc)
    return vec

# Initialise X (matrix of feature vectors) for train dataset
x_train = np.array([to_feature_array(doc) for doc in tqdm(train_all)])



# Define label <-> int mappings for y
label_to_int = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}
int_to_label = ['agree', 'disagree', 'discuss', 'unrelated']

# Initialise Y (gold output vector) for train dataset
y_train = np.array([label_to_int[i] for i in train_all[:, 2]])

print("over-sampling.")
sm = SMOTE(random_state=0)
# x_train, y_train = sm.fit_resample(x_train, y_train)
# print(len(x_train), len(y_train))

'----------------------------------plot feature relations-------------------------------------------'
# # Plot GLoVe distance vs KL-Divergence on a coloured scatter plot with different colours for each label
# colours = np.array(['g', 'r', 'b', 'y'])
# plt.scatter(list(x_train[:, 0]), list(x_train[:, 1]), c=colours[y_train])
# plt.xlabel('Cosine Similarity of GLoVe vectors')
# plt.ylabel('KL Divergence of Unigram LMs')
# print([(colours[i], int_to_label[i]) for i in range(len(int_to_label))])
# plt.show()


# Initialise x (feature vectors) for validation dataset
x_val = np.array([to_feature_array(doc) for doc in tqdm(val_all)])

# Linear regression model
def mse(pred, gold):
    sigma = 0.0
    for i in range(pred.shape[0]):
        sigma += np.square(pred[i] - gold[i])
    return sigma / (2 * pred.shape[0])

class LinearRegression:
    
    def __init__(self, lrn_rate, n_iter):
        self.lrn_rate = lrn_rate
        self.n_iter = n_iter
        # self.breakpoints = set([n_iter * i // 10 for i in range(1, 11)])
        
    def fit(self, X, Y):
        # Learn a model y = intercept + x0*t0 + x1*t1 + x2*t2 + ... that minimises MSE. Need to optimise T
        # self.intercept = 0.0
        self.model = np.zeros(X.shape[1])  # model[0] = t0, model[1] = t1, etc.
        for it in tqdm(range(self.n_iter)):
            model_Y = self.transform(X)
            # Thetas
            for col in range(X.shape[1]):
                s = 0.0
                for row in range(X.shape[0]):
                    s += (model_Y[row] - Y[row]) * X[row, col]
                self.model[col] -= self.lrn_rate * s / X.shape[0]
            # Intercept
            # s_int = 0.0
            # for row in range(X.shape[0]):
            #     s_int += (model_Y[row] - Y[row]) * 1.0
            # self.intercept -= self.lrn_rate * s_int / X.shape[0]
            # if it + 1 in self.breakpoints:
                # print('Iteration', it+1, 'MSE:', mse(model_Y, Y))
        print('Final MSE:', mse(model_Y, Y))
        # print('Intercept:', self.intercept)
        print('Model:', self.model)
        
    def transform(self, X):
        # Returns a float value for each X. (Regression)
        Y = np.zeros(X.shape[0])
        for row in range(X.shape[0]):
            # s = self.intercept
            s = 0.0
            for col in range(X.shape[1]):
                s += self.model[col] * X[row, col]
            Y[row] = s
        return Y
    
    def predict(self, X):
        # Uses results of transform() for binary classification. For testing only, use OneVAllClassifier for the final run.
        Y = self.transform(X)
        Y = np.array([(1 if i > 0.5 else 0) for i in Y])
        return Y

# Test only
lr = LinearRegression(lrn_rate=0.1, n_iter=100)
lr.fit(x_train[:1000], np.array([(1 if i == 3 else 0) for i in y_train[:1000]]))
print(lr.transform(x_train[1000:1020]))
print('Predicted', lr.predict(x_train[1000:1020]))
print('Actual', np.array([(1 if i == 3 else 0) for i in y_train[1000:1020]]))


# Logistic regression functions
def sigmoid(Y):
    return 1 / (1 + np.exp(Y * -1))


def logistic_cost(pred, gold):
    sigma = 0.0
    for i in range(pred.shape[0]):
        if gold[i] == 1:  
            sigma -= np.log(pred[i])
        elif gold[i] == 0:
            sigma -= np.log(1 - pred[i])
    return sigma / pred.shape[0]

# Logistic regression model
class LogisticRegression:
    
    def __init__(self, lrn_rate, n_iter):
        self.lrn_rate = lrn_rate
        self.n_iter = n_iter
        # self.breakpoints = set([n_iter * i // 10 for i in range(1, 11)])
        
    def fit(self, X, Y):
        # Learn a model y = x0*t0 + x1*t1 + x2*t2 + ... that minimises MSE. Need to optimise T
        self.model = np.zeros(X.shape[1])  # model[0] = t0, model[1] = t1, etc.
        for it in tqdm(range(self.n_iter)):
            model_Y = self.transform(X)
            for col in range(X.shape[1]):
                s = 0.0
                for row in range(X.shape[0]):
                    s += (model_Y[row] - Y[row]) * X[row, col]
                self.model[col] -= self.lrn_rate * s / X.shape[0]
            # if it + 1 in self.breakpoints:
                # print('Iteration', it+1, 'loss:', logistic_cost(model_Y, Y))
        print('Final loss:', logistic_cost(model_Y, Y))
        print('Model:', self.model)
        
    def transform(self, X):
        # Returns a float value for each X. (Regression)
        Y = np.zeros(X.shape[0])
        for row in range(X.shape[0]):
            s = 0.0
            for col in range(X.shape[1]):
                s += self.model[col] * X[row, col]
            Y[row] = s
        return sigmoid(Y)
    
    def predict(self, X):
        # Uses results of transform() for binary classification. For testing only, use OneVAllClassifier for the final run.
        Y = self.transform(X)
        Y = np.array([(1 if i > 0.5 else 0) for i in Y])
        return Y

# Test only
lr = LogisticRegression(lrn_rate=0.1, n_iter=100)
lr.fit(x_train[:1000], np.array([(1 if i == 3 else 0) for i in y_train[:1000]]))
print(lr.transform(x_train[1000:1020]))
print('Predicted', lr.predict(x_train[1000:1020]))
print('Actual', np.array([(1 if i == 3 else 0) for i in y_train[1000:1020]]))


# In[25]:


# To use linear/logistic regression models to classify multiple classes
class OneVAllClassifier:
    
    def __init__(self, regression, **params):
        self.regression = regression
        self.params = params
        
    def fit(self, X, Y):
        # Learn a model for each parameter.
        self.categories = np.unique(Y)
        self.models = {}
        for cat in self.categories:
            ova_Y = np.array([(1 if i == cat else 0) for i in Y])
            model = self.regression(**self.params)
            model.fit(X, ova_Y)
            self.models[cat] = model
            print(int_to_label[cat])
    
    def predict(self, X):
        # Predicts each x for each different model learned, and returns the category related to the model with the highest score.
        vals = {}
        for cat, model in self.models.items():
            vals[cat] = model.transform(X)
        Y = np.zeros(X.shape[0], dtype=np.int)
        for row in range(X.shape[0]):
            max_val, max_cat = -math.inf, -math.inf
            for cat, val in vals.items():
                if val[row] > max_val:
                    max_val, max_cat = val[row], cat
            Y[row] = max_cat
        return Y
    
# Test only
ova = OneVAllClassifier(LinearRegression, lrn_rate=0.1, n_iter=100)
ova.fit(x_train[:1000], y_train[:1000])
print('Predicted', ova.predict(x_train[1000:1020]))
print('Actual', y_train[1000:1020])


# In[26]:

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
# Train the linear regression & One-V-All classifier models on the train set
# clf = OneVAllClassifier(LinearRegression, lrn_rate=0.1, n_iter=1000)
# clf = LinearRegression()
clf = SVC(C=50.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf.fit(x_train, y_train)


# In[27]:


# Predict y for validation set
y_pred = clf.predict(x_val)
print(y_pred[:5])
predicted = np.array([int_to_label[i] for i in y_pred])
print(predicted[:5])
print(val_all[:, 2][:5])


# In[28]:


# Prepare validation dataset format for score_submission in scorer.py
body_ids = [str(body_inverse_index[body]) for body in val_all[:, 1]]
pred_for_cm = np.array([{'Headline': val_all[i, 0], 'Body ID': body_ids[i], 'Stance': predicted[i]} for i in range(len(val_all))])
gold_for_cm = np.array([{'Headline': val_all[i, 0], 'Body ID': body_ids[i], 'Stance': val_all[i, 2]} for i in range(len(val_all))])


# In[29]:


# Score using scorer.py (provided in https://github.com/FakeNewsChallenge/fnc-1) on VALIDATION set:
test_score, cm = score_submission(gold_for_cm, pred_for_cm)
null_score, max_score = score_defaults(gold_for_cm)
print_confusion_matrix(cm)
print(SCORE_REPORT.format(max_score, null_score, test_score))


# In[30]:


# Predict y for validation set using logistic regression instead of linear regression, and compare results of scorer.py
clf_logistic = OneVAllClassifier(LogisticRegression, lrn_rate=0.1, n_iter=1000)
clf_logistic.fit(x_train, y_train)

y_pred = clf_logistic.predict(x_val)
predicted = np.array([int_to_label[i] for i in y_pred])

body_ids = [str(body_inverse_index[body]) for body in val_all[:, 1]]
pred_for_cm = np.array([{'Headline': val_all[i, 0], 'Body ID': body_ids[i], 'Stance': predicted[i]} for i in range(len(val_all))])
gold_for_cm = np.array([{'Headline': val_all[i, 0], 'Body ID': body_ids[i], 'Stance': val_all[i, 2]} for i in range(len(val_all))])

test_score, cm = score_submission(gold_for_cm, pred_for_cm)
null_score, max_score = score_defaults(gold_for_cm)
print()
print_confusion_matrix(cm)
print(SCORE_REPORT.format(max_score, null_score, test_score))
# linear regression performs better, so that model is chosen for the test set


# In[31]:


# Load test data from CSV
f_tbodies = open('data/test_bodies.csv', 'r', encoding='utf-8')
csv_tbodies = csv.DictReader(f_tbodies)
tbodies = []
for row in csv_tbodies:
    body_id = int(row['Body ID'])
    if (body_id + 1) > len(tbodies):
        tbodies += [None] * (body_id + 1 - len(tbodies))
    tbodies[body_id] = row['articleBody']
f_tbodies.close()
tbody_inverse_index = {tbodies[i]: i for i in range(len(tbodies))}

test_all = []  # each article = (headline, body, stance)

f_tstances = open('data/competition_test_stances.csv', 'r', encoding='utf-8')
csv_tstances = csv.DictReader(f_tstances)
for row in csv_tstances:
    body = tbodies[int(row['Body ID'])]
    test_all.append((row['Headline'], body, row['Stance']))
f_tstances.close()

#test_all = np.array(test_all)  # for some reason gives MemoryError
#print(test_all.shape)


# In[32]:


# Initialise x (feature vectors) and y for test dataset
x_test = np.array([to_feature_array(doc) for doc in tqdm(test_all)])
# print(x_test[:5])


# In[33]:


# Predict y for test set
y_test = clf.predict(x_test)
# print(y_pred[:5])
pred_test = np.array([int_to_label[i] for i in y_test])
print(pred_test[:5])


# In[34]:


# Prepare test dataset format for score_submission in scorer.py
test_body_ids = [str(tbody_inverse_index[test_all[i][1]]) for i in range(len(test_all))]
test_pred_for_cm = np.array([{'Headline': test_all[i][0], 'Body ID': test_body_ids[i], 'Stance': pred_test[i]} for i in range(len(test_all))])
test_gold_for_cm = np.array([{'Headline': test_all[i][0], 'Body ID': test_body_ids[i], 'Stance': test_all[i][2]} for i in range(len(test_all))])


# In[35]:


# Score using scorer.py (provided in https://github.com/FakeNewsChallenge/fnc-1) on TEST set:
test_score, cm = score_submission(test_gold_for_cm, test_pred_for_cm)
null_score, max_score = score_defaults(test_gold_for_cm)
print_confusion_matrix(cm)
print(SCORE_REPORT.format(max_score, null_score, test_score))
print(test_pred_for_cm.shape)
print(test_gold_for_cm.shape)

# write answer.csv
from utils import utils
competition_test = utils.read('competition_test_stances.csv')
print(len(competition_test), len(y_test))
utils.write_out(competition_test, y_test)
exit()

# ## Comparison Models

# In[36]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier   
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import xgboost
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.semi_supervised import LabelPropagation

classifiers=[
    
    (LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=100.0, fit_intercept=True,
    intercept_scaling=10, class_weight=None, random_state=None, solver='warn', max_iter=10,
    multi_class='warn', verbose=0, warm_start=False, n_jobs=None),"Logistic Regression"),
    
    
    (KNeighborsClassifier(1),"K Nearest Classifier "),
    
    (SVC(C=50.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False),'Support Vector Machine Classifier'),
    
    (QuadraticDiscriminantAnalysis(),'Qudratic Discriminant Analysis'),
    
    (RandomForestClassifier(max_depth=50, n_estimators=10, max_features=1),'Random Forest Classifier'),
    
    (AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=0.01,
                        algorithm='SAMME.R', random_state=None),'Adaboost Classifier'),
    (SGDClassifier(),'SGD Classifier'),
    
    (DecisionTreeClassifier(max_depth=5),'Decision Tree Classifier'),
    (xgboost.XGBClassifier(learning_rate=0.1),'XG Boost Classifier'),
    
    (LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, n_components=None, 
        store_covariance=False,tol=0.00001),'Linear Discriminant Analysis'),
     
    (GaussianNB(),'Gaussian Naive Bayes ')
]


# In[37]:


score=[];names=[]

for model,name in classifiers:
    clf=model.fit(x_train,y_train)
    y_pred=clf.predict(x_val)
    predicted = np.array([int_to_label[i] for i in y_pred])
    body_ids = [str(body_inverse_index[body]) for body in val_all[:, 1]]
    pred_for_cm = np.array([{'Headline': val_all[i, 0], 'Body ID': body_ids[i], 'Stance': predicted[i]} for i in range(len(val_all))])
    gold_for_cm = np.array([{'Headline': val_all[i, 0], 'Body ID': body_ids[i], 'Stance': val_all[i, 2]} for i in range(len(val_all))])
    
    test_score, cm = score_submission(gold_for_cm, pred_for_cm)
    null_score, max_score = score_defaults(gold_for_cm)
    print('*'*20);names.append(name)
    print(name)
    score.append(print_confusion_matrix(cm));
    a=SCORE_REPORT.format(max_score, null_score, test_score)


# In[38]:


for i in range(len(score)):
    score[i]=score[i]*100
print(score)


# In[39]:


names=['LR','KNC','SVC','QDA','RFC','ADC','SGDC','DTC','XGB','LDA','GNB']
import seaborn as sns
from matplotlib import pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
A = score[:]
plt.plot(A)
for i, label in enumerate(names):
    plt.text(i,A[i], label)
plt.show()


# In[ ]:





# In[ ]:




