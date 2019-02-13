__author__ = 'yshi31'
from pandas import *
# load data
from ConfigParser import SafeConfigParser
from sklearn.feature_extraction.text import TfidfVectorizer

parser = SafeConfigParser()
parser.read('config.ini')
LABEL_FOLDER = parser.get('BASIC', 'LABEL_FOLDER')
RESULT_FOLDER = parser.get('BASIC', 'RESULT_FOLDER')

class tweetdata:
    def __init__(self,president,file_name):
        self.president = president
        self.data = read_csv(LABEL_FOLDER + file_name,header= None,low_memory=False)
        print self.data.shape
        self.data.columns = ['date','time','textbody','class']
        # print self.data.groupby('class').count()
        #clean the data
        self.data = self.data[np.isfinite(self.data['class'])]
        self.data = self.data[self.data['class'] != 2]
        self.data = self.data[self.data['textbody'] != '']


        #decode the data
    def decode(self,row,var):
        return str(row[var]).lower().decode('utf-8', 'ignore')

    def vectorize(self):
        corpus = self.data.apply(lambda row:self.decode(row,'textbody'),axis=1)
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        y = self.data['class'].tolist()
        y = [int(i) for i in y]
        return X,y