from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
s = np.array([1,2,3])
a = np.array([4,5])
print(np.concatenate((s, a)))
exit()
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
vectorizer = TfidfVectorizer(max_features=5)
X = vectorizer.fit_transform(corpus)
print(X.toarray())
print(vectorizer.get_feature_names())

print(X.shape)

