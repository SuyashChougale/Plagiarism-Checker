suy = open(r"suyash.txt","r",errors="ignore")
vai = open(r"Vaibhav.txt","r",errors="ignore")

df1 = suy.read()
df2 = vai.read()

import string

from nltk.tokenize import word_tokenize,sent_tokenize

sents1 = sent_tokenize(df1)
sents2 = sent_tokenize(df2)
from nltk.stem import PorterStemmer,WordNetLemmatizer

stemmer = PorterStemmer()
lemmer = WordNetLemmatizer()

for i in range(len(sents1)):
    sents1[i] = sents1[i].translate(str.maketrans('','',string.punctuation))
    words = word_tokenize(sents1[i])
    for j in range(len(words)):
        words[j] = stemmer.stem(words[j])
        words[j] = lemmer.lemmatize(words[j])
    sents1[i] = ' '.join(words) 
for i in range(len(sents2)):
    sents2[i] = sents2[i].translate(str.maketrans('','',string.punctuation))
    words = word_tokenize(sents2[i])
    for j in range(len(words)):
        words[j] = stemmer.stem(words[j])
        words[j] = lemmer.lemmatize(words[j])
    sents2[i] = ' '.join(words) 
# df1 = ' '.join(sents1)
# df2 = ' '.join(sents2)
##########################################################
#countvectorizer
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
vec1 = vec.fit_transform(sents1,sents2)

from sklearn.metrics.pairwise import cosine_similarity
score = cosine_similarity(vec1[0],vec1[1])

print(score)
##########################################################
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
vec2 = tfidf.fit_transform(sents1,sents2)
# vec2 = tfidf.fit_transform(sents2)
score2 = cosine_similarity(vec2[0],vec2[1])
# num_pairs = (len(score2))*(len(score2)-1)/2
# avg = score2.sum()/num_pairs
print(score2)
# *************************************************************
from sklearn.metrics.pairwise import cosine_distances
# vec1 = vec.fit_transform(sents1)
# vec2 = vec.fit_transform(sents2)

dist2 = cosine_distances(vec2[0],vec2[1])
dist1 = cosine_distances(vec1[0],vec1[1])
print(dist1)
print(dist2)