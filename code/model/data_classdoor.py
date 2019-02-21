__author__ = 'Xuelong'
from pandas import *
from numpy import *
# load data
from ConfigParser import SafeConfigParser
parser = SafeConfigParser()
parser.read('config.ini')
LABEL_FOLDER = parser.get('BASIC', 'LABEL_FOLDER')
RESULT_FOLDER = parser.get('BASIC', 'RESULT_FOLDER')

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

class glassdoor:
    def __init__(self,file_name):
        # load data from csv files
        print "loading the apply data..."
        self.data = read_csv(LABEL_FOLDER + file_name, low_memory=False)
        print "original data size", self.data.shape 
        # drop all NaNs
        print "drop all the NA rows..."
        self.data = self.data.dropna(axis=0, how="any")
        print "clean data size", self.data.shape
        # covert the time columns
        self.data["search_date_pacific"] = to_datetime(self.data["search_date_pacific"])
        print "apply rate", self.data['apply'].mean().round(3)

    def scale_data(self, method = "Max_Mix"):
        print "summary before the scaling\n", self.data.describe()
        if(method == "Max_Mix"):
            scale_feature = scaler.fit_transform(self.data.iloc[:,0:-3])
        self.data.iloc[:,0:-3] = scale_feature
        print "summary after the scaling\n", self.data.describe()

    def split_data(self):
        date1 = to_datetime("2018-01-21")
        date2 = to_datetime("2018-01-26")
        date_test = to_datetime("2018-01-27")
        Train_data = self.data.loc[(self.data["search_date_pacific"] >= date1) & (self.data["search_date_pacific"] <= date2)]
        Test_data = self.data.loc[(self.data["search_date_pacific"] == date_test)]
        return Train_data, Test_data
