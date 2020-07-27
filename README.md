<p align="center">
<img src="https://github.com/jimjimliu/Fake-News-Stance-Detection/blob/master/img/fnc.png" alt="fnc" width="25%"/>
<img src="https://github.com/jimjimliu/Fake-News-Stance-Detection/blob/master/img/uw.png" alt="uw" width="35%"/>
</p>


<h1 align = "center">UWaterloo - Fake News Challenge</h1>
<p align="center">
FAKE NEWS CHALLENGE STAGE 1 (FNC-I): 
STANCE DETECTION
</p>


<h2>Introduction</h2>
<blockquote><p><strong>Input</strong></p>
<p style="text-indent:2em">&emsp;&emsp;A headline and a body text - either from the same news article or from two different articles.</p>
</blockquote>
<blockquote><p><strong>Output</strong></p>
<p style="text-indent:2em">&emsp;&emsp;Classify the stance of the body text relative to the claim made in the headline into one of four categories:</p>
<p style="text-indent:4em">&emsp;&emsp;&emsp;&emsp;<strong>Agrees</strong>: The body text agrees with the headline.</p>
<p style="text-indent:4em">&emsp;&emsp;&emsp;&emsp;<strong>Disagrees</strong>: The body text disagrees with the headline.</p>
<p style="text-indent:4em">&emsp;&emsp;&emsp;&emsp;<strong>Discusses</strong>: The body text discuss the same topic as the headline, but does not take a position</p>
<p style="text-indent:4em">&emsp;&emsp;&emsp;&emsp;<strong>Unrelated</strong>: The body text discusses a different topic than the headline</p>
</blockquote>
<p>&nbsp;</p>

Stance Detection involves estimating the relative perspective (or stance) of two pieces of text relative to a topic, claim or issue. For FNC-1 we have chosen the task of estimating the stance of a body text from a news article relative to a headline. Specifically, the body text may **agree, disagree, discuss** or be **unrelated** to the headline.

*For details of the task, see [FakeNewsChallenge.org](http://fakenewschallenge.org)*



------

### Approach

The general approach here is to use a combination of conventional classifier and feed forward neural network model to tegether classify future instances. The choice of a conventional classifier chosen is SVM, which is used to do a binary classification to predict if an instance is **related** or **unrelated** to a headline. The neural network model is classfiying if an instance is one of the following **agree, disagree, discuss**. 

Three kinds of features are extracted between a headline and body article: `cosine similarity, kl-divergence, n-gram overlap` 

The two classifiers are trained using the same data set, but under different target labels. For SVM, the entire the target labels are changed to either related or unrelated. For neural network training input, labels have values of agree, disagree, discuss are keeped. 

When predicting(tesing), the same data set is used. For instances that are marked as **unrelated** by SVM, they are keeped untouch. For those which are labeld as **related**, their target labels are swaped by the predictions(**agree, disagree, discuss**) of neural network. 



<p align="center">
<img src="https://github.com/jimjimliu/Fake-News-Stance-Detection/blob/master/img/mind.png" alt="fnc" width="100%"/>
</p>


`

The result of using 2-step classification is as follows:

```markdown
ACCURACY: 0.865

MAX  - the best possible score (100% accuracy)
NULL - score as if all predicted stances were unrelated
TEST - score based on the provided predictions

||    MAX    ||    NULL   ||    TEST   ||
|| 11651.25  ||  4587.25  ||  9055.5   ||
```

The average accuracy reached is 86.5%, and [leader board](https://competitions.codalab.org/competitions/16843#results) score is 9055.5.

### System Requirement

```
Python 3.7.4
```

### Brief

The source code contains two main .py files: 1. `main.py`, 2. `FeatureExtract.py`.

The first file is the driver script which contains the entire life cycle of the classification process; it calls according APIs to get cleaned data -> extract features out of headlines and bodies -> feed the models for classification.

The latter one contians the life cycle of taking in headline and body as input and output features.

The other .py files are helper scripts.

### To Run

`main.py` is the driver script. Simply run `main.py` to re-produce submission(prediction) files stored in `../data/submission/` folder.

All data files used are provided by [fnc-1](https://github.com/FakeNewsChallenge/fnc-1) on GitHub, those files are put in `../data/` folder. 

Before running, download word embedding file [glove.6B.50d.txt](https://www.kaggle.com/pkugoodspeed/nlpword2vecembeddingspretrained?select=glove.6B.50d.txt), put it into `../data/` folder. You are good to go.
