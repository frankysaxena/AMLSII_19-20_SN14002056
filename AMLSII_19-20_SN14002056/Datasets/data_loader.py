import pandas as pd 

class DataPreprocessor:

    def __init__(self, path, txtfile):

        self.path = path
        self.txtfile = txtfile

    def get_raw_dataframe(self, path, txtfile):
        
        """ Using a simple pandas read_csv import to extract labels from the dataset """
        
        return pd.read_csv(self.path + '/Datasets/' + self.txtfile  + '.csv', sep=',', header=None)

