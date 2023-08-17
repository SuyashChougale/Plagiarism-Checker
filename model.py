from nltk.stem import PorterStemmer,WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize,sent_tokenize
import string


suy = open(r"suyash.txt","r",errors="ignore")
vai = open(r"Vaibhav.txt","r",errors="ignore")
# Reading the 2 required files for comparison
df1 = suy.read()
df2 = vai.read()

# Tokenizing text into sentences
sents1 = sent_tokenize(df1)
sents2 = sent_tokenize(df2)

#importing tools for stemming and lemmatization
stemmer = PorterStemmer()
lemmer = WordNetLemmatizer()
# preprocessing each sentence of first file
for i in range(len(sents1)):
    # removing the punctuations from the sentences
    sents1[i] = sents1[i].translate(str.maketrans('','',string.punctuation))
    # tokenizing sentence into words
    words = word_tokenize(sents1[i])

    for j in range(len(words)):
        # stemming and lemmatizing each word of sentence
        words[j] = stemmer.stem(words[j])
        words[j] = lemmer.lemmatize(words[j])
    # joining the stemmed and lemmatized words to form the sentence again
    sents1[i] = ' '.join(words) 


# preprocessing each sentence of second file
for i in range(len(sents2)):
    # removing the punctuations from the sentences
    sents2[i] = sents2[i].translate(str.maketrans('','',string.punctuation))
    # tokenizing sentence into words
    words = word_tokenize(sents2[i])
    for j in range(len(words)):
        # stemming and lemmatizing each word of sentence
        words[j] = stemmer.stem(words[j])
        words[j] = lemmer.lemmatize(words[j])
    # joining the stemmed and lemmatized words to form the sentence again
    sents2[i] = ' '.join(words) 

# Method 1
#importing countvectorizer to vectorize the sentences

vec = CountVectorizer()
# storing vectors for each sentence 
vec1 = vec.fit_transform(sents1,sents2)

# checking similarity between both texts
score = cosine_similarity(vec1[0],vec1[1])
dist1 = cosine_distances(vec1[0],vec1[1])
print(score)
print(dist1)

# method 2

tfidf = TfidfVectorizer()
# storing vectors for each sentence 
vec2 = tfidf.fit_transform(sents1,sents2)

# checking similarity between both texts
score2 = cosine_similarity(vec2[0],vec2[1])
dist2 = cosine_distances(vec2[0],vec2[1])
print(score2)
print(dist2)

