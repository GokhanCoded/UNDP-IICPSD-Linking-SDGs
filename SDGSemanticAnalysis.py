#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:04:19 2019

@author: jonas
"""

"""
1. Clear all variables
"""

def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]
if __name__ == "__main__":
    clear_all()

"""
2. Import packages 
"""

from collections import OrderedDict
import sys
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en.examples import sentences

import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from string import punctuation
from collections import Counter
import mpld3


from collections import OrderedDict
import re
import warnings
import itertools as it

warnings.filterwarnings("ignore", category=DeprecationWarning)

from html.parser import HTMLParser
from bs4 import BeautifulSoup

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
    
porter = PorterStemmer()
wnl = WordNetLemmatizer()
stop = stopwords.words('english')
stop.append("new")
stop.append("like")
stop.append("u")
stop.append("it'")
stop.append("'s")
stop.append("n't")
stop.append('mr.')
stop = set(stop)

porter = PorterStemmer()
wnl = WordNetLemmatizer()
stop = stopwords.words('english')
stop.append("new")
stop.append("like")
stop.append("u")
stop.append("it'")
stop.append("'s")
stop.append("n't")
stop.append('mr.')
stop = set(stop)

"""
Import Data
"""
#First, we import the three schemes respectively:
#import first sheet: SDG Goals
sdg = pd.read_excel('sdg.xlsx', header=None, encoding='latin', skip_blank_lines=False)
sdg.fillna(0, inplace=True)
sdg = sdg.where(sdg != "x", 1)
sdg = sdg.where(sdg != "X", 1)

#import second sheet: SDG Targets
target = pd.read_excel('Target.xlsx', header=None, encoding='latin', skip_blank_lines=False)
target.fillna(0, inplace=True)
target = target.where(target != "x", 1)
target = target.where(target != "X", 1)

#import third sheet: SDG Indicators
indicator = pd.read_excel('Indicator.xlsx', header=None, encoding='latin', skip_blank_lines=False)
indicator.fillna(0, inplace=True)
indicator = indicator.where(indicator != "x", 1)
indicator = indicator.where(indicator != "X", 1)

keys = sdg.iloc[0,2:1317] #could have used also indicators or targets to obtain keys
keys = keys.astype(str)
keys = keys.reset_index()
keys = keys.drop(['index'], axis=1)
#decapitalize keywords for better matching
a = keys[0].tolist()
a = [element.lower() for element in a]
strip = pd.DataFrame({0:a})

#because indicator has one key word more we use also the indicator keywords.
keys2 = indicator.iloc[0,4:1319]
keys2 = keys2.astype(str)
keys2 = keys2.reset_index()
keys2 = keys2.drop(['index'], axis=1)
a = keys[0].tolist()
a = [element.lower() for element in a]
strip2 = pd.DataFrame({0:a})

"""
Goals
"""
sdg.columns = sdg.iloc[0]
goals = sdg.iloc[1:18,0:2]
goals = goals.astype(str)
goals = goals.reset_index()
goals = goals.drop(['index'], axis=1)

#decapitalize goals for better matching
a = goals['Description'].tolist()
a = [element.lower() for element in a]
goals_lower = pd.DataFrame({'Description':a})
del goals['Description']
frames = [goals, goals_lower]
goals = pd.concat(frames, axis=1)

"""
Targets
"""
target.columns = target.iloc[0]
targets = target.iloc[1:170,0:3]
targets = targets.astype(str)
targets = targets.reset_index()
targets = targets.drop(['index'], axis=1)

#decapitalize targets for better matching
a = targets['Description'].tolist()
a = [element.lower() for element in a]
target_lower = pd.DataFrame({'Description':a})
del targets['Description']
frames = [targets, target_lower]
targets = pd.concat(frames, axis=1)

"""
Indicators
"""
indicator.columns = indicator.iloc[0]
indicators = indicator.iloc[1:246,0:4]
indicators = indicators.astype(str)
indicators = indicators.reset_index()
indicators = indicators.drop(['index'], axis=1)

#decapitalize indicators for better matching
a = indicators['Description'].tolist()
a = [element.lower() for element in a]
indicator_lower = pd.DataFrame({'Description':a})
del indicators['Description']
frames = [indicators, indicator_lower]
indicators = pd.concat(frames, axis=1)

"""
Goals + Targets + Indicators
"""
#group by goals and sum up string in targets and indicators
target_sum = targets.groupby(['Goal'],as_index=False).agg(lambda x : x.sum() if x.dtype=='float64' else ' '.join(x))
indicator_sum = indicators.groupby(['Goal'],as_index=False).agg(lambda x : x.sum() if x.dtype=='float64' else ' '.join(x))

goals['target'] = target_sum['Description']
goals['indicator'] = indicator_sum['Description']
#rename SDG
goals = goals.rename(index=str, columns={'Descript6ion' : "SDG"})
goals['total']= goals.stack().groupby(level=0).apply(', '.join)

"""
Include stopwords, tokenize data
"""

def tokenizer(text):
    tokens_ = [word_tokenize(sent) for sent in sent_tokenize(text)]

    tokens = []
    for token_by_sent in tokens_:
        tokens += token_by_sent

    tokens = list(filter(lambda t: t.lower() not in stop, tokens))
    tokens = list(filter(lambda t: t not in punctuation, tokens))
    tokens = list(
        filter(lambda t: t not in [u"'s", u"n't", u"...", u"''", u'``', u'\u2014', u'\u2026', u'\u2013'], tokens))

    filtered_tokens = []
    for token in tokens:
        token = wnl.lemmatize(token)
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)

    filtered_tokens = list(map(lambda token: token.lower(), filtered_tokens))

    return filtered_tokens

class MLStripper(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

# find top keywords (most mentioned)
def get_keywords(tokens, num):
    return Counter(tokens).most_common(num)


def build_article_df(urls):
    articles = []
    for index, row in urls.iterrows():
        try:
            data = row['total'].strip().replace("'", "")
            data = strip_tags(data)
            soup = BeautifulSoup(data, 'html.parser')
            data = soup.get_text()
            data = data.encode('ascii', 'ignore').decode('ascii')
            document = tokenizer(data)
            top_5 = get_keywords(document, 10)

            unzipped = list(zip(*top_5))
            kw = list(unzipped[0])
            kw = ",".join(str(x) for x in kw)
            articles.append((kw, row['total']))
        except Exception as e:
            print(e)
            # print data
            # break
            pass
        # break
    article_df = pd.DataFrame(articles, columns=['keywords', 'total'])
    return article_df

data = goals['total']
data = data.to_frame()
article_df = build_article_df(data)


"""
Analysis - working with own keywords
"""

"""
1. Co-occurence matrix with keywords obtained by highest number of matches (Network map)
"""
keywords_array=[]
for index, row in article_df.iterrows():
    keywords=row['keywords'].split(',')
    for kw in keywords:
        keywords_array.append((kw.strip(' '), row['keywords']))

kw_df = pd.DataFrame(keywords_array).rename(columns={0:'keyword', 1:'keywords'})

document = kw_df.keywords.tolist()
names = kw_df.keyword.tolist()

document_array = []
for item in document:
    items = item.split(',')
    document_array.append((items))

occurrences = OrderedDict((name, OrderedDict((name, 0) for name in names)) for name in names)

for l in document_array:
    for i in range(len(l)):
        for item in l[:i] + l[i + 1:]:
            occurrences[l[i]][item] += 1

co_occur = pd.DataFrame.from_dict(occurrences)

co_occur.to_csv('Co-ocurrence_topKW.csv')

"""
2.. Count Matches
"""
res = []
for a in article_df['total']:
    res.append(strip2.applymap(lambda x: str.count(a, x)))

map = pd.concat(res, axis=1).T
map.index = np.arange(len(map))
map.columns = strip2
cor = map.corr()
# delete columns with zero matches
map = map.loc[:, (map != 0).any(axis=0)]
map.to_csv('correlation_mapping.csv')

"""
3. Co-occurence matrix with own keywords from above
"""
from sklearn.feature_extraction.text import CountVectorizer

samples = article_df['total'].tolist()
keywords = strip2[0].tolist()

bigram_vectorizer = CountVectorizer(ngram_range=(1,2), vocabulary = keywords)
X = bigram_vectorizer.fit_transform(samples)
Xc = (X.T * X)
Xc.setdiag(0)
names = bigram_vectorizer.get_feature_names()
biagram = pd.DataFrame(data = Xc.toarray(), columns = names, index = names)
'''good idea??'''
biagram = biagram.loc[:, (biagram != 0).any(axis=0)]

biagram.to_csv('co_occurenceSDG.csv', sep = ',')
'''Problem: we have way to many meaningless keywords to get a clear picture.'''

"""
4. Apply TextRank to obtain keywords by unsupervised ML (Python 3 needed now)
"""
nlp = spacy.load('en_core_web_sm')

__builtins__.list()
class TextRank4Keyword():
    """Extract keywords from text"""

    def __init__(self):
        self.d = 0.85  # damping coefficient, usually is .85
        self.min_diff = 1e-5  # convergence threshold
        self.steps = 10  # iteration steps
        self.node_weight = None  # save keywords and its weight

    def set_stopwords(self, stopwords):
        """Set stop words"""
        for word in STOP_WORDS.union(set(stopwords)):
            lexeme = nlp.vocab[word]
            lexeme.is_stop = True

    def sentence_segment(self, doc, candidate_pos, lower):
        """Store those words only in cadidate_pos"""
        sentences = []
        for sent in doc.sents:
            selected_words = []
            for token in sent:
                # Store words only with cadidate POS tag
                if token.pos_ in candidate_pos and token.is_stop is False:
                    if lower is True:
                        selected_words.append(token.text.lower())
                    else:
                        selected_words.append(token.text)
            sentences.append(selected_words)
        return sentences

    def get_vocab(self, sentences):
        """Get all tokens"""
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab

    def get_token_pairs(self, window_size, sentences):
        """Build token_pairs from windows in sentences"""
        token_pairs = list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i + 1, i + window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs

    def symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())

    def get_matrix(self, vocab, token_pairs):
        """Get normalized matrix"""
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1

        # Get Symmeric matrix
        g = self.symmetrize(g)

        # Normalize matrix by column
        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, where=norm != 0)  # this is ignore the 0 element in norm

        return g_norm

    def get_keywords(self, number=30):
        """Print top number keywords"""
        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
        for i, (key, value) in enumerate(node_weight.items()):
            print(key + ' - ' + str(value))
            if i > number:
                break

    def analyze(self, text,
                candidate_pos=['NOUN', 'PROPN'],
                window_size=4, lower=False, stopwords=list()):
        """Main function to analyze text"""

        # Set stop words
        self.set_stopwords(stopwords)

        # Pare text by spaCy
        doc = nlp(text)

        # Filter sentences
        sentences = self.sentence_segment(doc, candidate_pos, lower)  # list of list of words

        # Build vocabulary
        vocab = self.get_vocab(sentences)

        # Get token_pairs from windows
        token_pairs = self.get_token_pairs(window_size, sentences)

        # Get normalized matrix
        g = self.get_matrix(vocab, token_pairs)

        # Initionlization for weight(pagerank value)
        pr = np.array([1] * len(vocab))

        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr = (1 - self.d) + self.d * np.dot(g, pr)
            if abs(previous_pr - sum(pr)) < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]

        self.node_weight = node_weight
        

text = article_df.total.tolist()
text = ''.join(text)

print('----------------------------------')
print('Ranked Keywords for SDG+Targets+Indicators:')
print('----------------------------------')
tr4w = TextRank4Keyword()
tr4w.analyze(text, candidate_pos = ['NOUN', 'PROPN'], window_size=4, lower=False)
tr4w.get_keywords(30)
