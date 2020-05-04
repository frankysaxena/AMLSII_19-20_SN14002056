import pandas as pd 

class DataPreprocessor:

    def __init__(self, path, txtfile):

        self.path = path
        self.txtfile = txtfile

    def get_raw_dataframe(self, path, txtfile):
        
        """ Using a simple pandas read_csv import to extract labels from the dataset """
        
        df = pd.read_csv(self.path + '/Datasets/' + self.txtfile  + '.csv', sep=',', header=None)
        message_df = A_df[2].values
        labels_df = A_df[1].values

        # A_df = A_df.rename(columns = {0: "id", 1: "label", 2: "message", 3: "nan"})

    def split_train_test(self, dataframe):
        
        """Taking the datasets that have been computed and then convert them into the respective numpy arrays"""
        message = self.dataframe['message'] 
        label = self.dataframe['label']
        
        """Split the data into testing and training sets"""
        """As mentioned, validation set is computed during cross-validation, GridSearch tasks"""

        x, x_test, y, y_test = train_test_split(
            message,
            label,
            test_size=0.2,
            shuffle=True,
            random_state=42,
        )
        
        train_data = (x, y)
        test_data = (x_test, y_test)
        
        return train_data, test_data
