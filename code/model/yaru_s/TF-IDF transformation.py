# TF-IDF transformation
# http://scikit-learn.org/stable/modules/feature_extraction.html
#
#
# transform txt data to matrix with counts for each word as values
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(min_df=1)
corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]
X = vectorizer.fit_transform(corpus)
X.toarray()
vectorizer.get_feature_names()
analyze = vectorizer.build_analyzer()
analyze("This is a text document to analyze.") == (
    ['this', 'is', 'text', 'document', 'to', 'analyze'])

# transform the matrix to tf-idf matrix
from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer(smooth_idf=True)

counts = X

tfidf = transformer.fit_transform(counts)
tfidf
tfidf.toarray()

# As tf–idf is very often used for text features, there is also another class called
# TfidfVectorizer that combines all the options of CountVectorizer and TfidfTransformer in a single model:

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(min_df=1)
x = vectorizer.fit_transform(corpus)

# the following matrix should be as same as the "tfidf.toarray()"

x.toarray()
