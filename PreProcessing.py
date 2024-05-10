import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
import pandas as pd
class Preprocessing():
    def __init__(self, df):
        self.df=df
    def standardScaler(self):
        standard_scaler = StandardScaler()
        for column in self.df.columns:
            if self.df[column].dtype in ['int64', 'float64']:
                self.df[column] = standard_scaler.fit_transform(self.df[[column]])
        return self.df
    def minMaxScaler(self):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(self.df)
        scaled_df = pd.DataFrame(scaled_data, columns=self.df.columns)
        return scaled_df

    def outlierRemoval(self, threshold=3):
        z_scores = stats.zscore(self.df.select_dtypes(include=['int64', 'float64']))

        outlier_indices = (np.abs(z_scores) > threshold).any(axis=1)

        df = self.df[~outlier_indices]
        return df

    def creatingSequences(self, data, sequence_length=400):
        '''
        We should reduce the leftover data by adjusting sequence_length to be the closest value to the passed sequence_length.
        Priority 2!!
        '''
        sequences = []
        for i in range(0, len(data) - sequence_length + 1, sequence_length):
            sequence = data[i:i + sequence_length]
            sequences.append(sequence)
        return np.array(sequences)

    def splitter(self, target, random_state, test_size):
        '''
        Just splits data using target and hyperparameters for more scalable code
        '''
        x = self.df.drop(columns=[target])
        y = self.df[target]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        return x_train, x_test, y_train, y_test

    def dateTimeFormater(self,df,columnName):
        df[columnName] = pd.to_datetime(df[columnName])
        df['year'] = df[columnName].dt.year
        df['month'] = df[columnName].dt.month
        df['day'] = df[columnName].dt.day
        df.drop(columns=[columnName], inplace=True)
        return df