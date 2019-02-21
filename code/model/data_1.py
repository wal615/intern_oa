__author__ = 'Xuelong'
from pandas import *
from numpy import *
# load data
from ConfigParser import SafeConfigParser
from sklearn.feature_extraction.text import TfidfVectorizer
# the current wd is '/Users/Ben/PycharmProjects/cs583/project2/model'
parser = SafeConfigParser()
parser.read('config.ini')
LABEL_FOLDER = parser.get('BASIC', 'LABEL_FOLDER')
RESULT_FOLDER = parser.get('BASIC', 'RESULT_FOLDER')

class tweetdata:
    def __init__(self,president,file_name):
        self.president = president
        # load data from csv files
        self.data = read_csv(LABEL_FOLDER + file_name, low_memory=False)
        print "original data size", self.data.shape
        self.data.columns = ['date','time','textbody','class']
        # convert "class" to numeric
        self.data["class"] = to_numeric(self.data["class"],errors="coerce")
        # drop all NaNs
        self.data = self.data.dropna(axis=0, how="any")
        # self.data = self.data[np.isfinite(self.data['class'])]
        self.data = self.data[self.data['class'] != 2]
        self.data = self.data[self.data['textbody'] != '']
        print "clean data size", self.data.shape
        print self.data.groupby('class').count()
        # modify the time and data
        # data.time.replace({"-05:00": ""}, regex=True)

        # decode the data
    def decode(self,row,var):
        return str(row[var]).lower().decode('utf-8', 'ignore')

    # def decode(row, var):
    #     return str(row[var]).lower().decode('utf-8', 'ignore')

    def vectorize(self):
        corpus = self.data.apply(lambda row: self.decode(row, 'textbody'), axis=1)
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        y = self.data['class']
        # y = self.data['class'].tolist()
        # y = [int(i) for i in y]
        y = array(y)
        return X, y

    # data.to_csv('../data/Obama_1_c.csv')
