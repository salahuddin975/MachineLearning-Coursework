import numpy as np
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from Constants import MissingData


class DataPreprocessing:
    def __init__(self):
        pass

    def preprocess_missing_value(self, df, processing_type):
        df.replace(' ?', np.NaN, inplace=True)           # replace ' ?' with standard np.nan

    #    print(df.isnull().sum())                         # Count missing values per attribute
    #    msno.matrix(df, figsize=(10, 6), fontsize=7)     # Plot of missing data
    #    plt.show()

        if (processing_type == MissingData.RemoveEntireExample):
            print("Remove all examples with missing value")
            df.dropna(inplace=True)
        if (processing_type == MissingData.ReplaceWithMostFrequentData):
            print("Replacing all missing value with most frequent value")
            df.fillna(df.mode().iloc[0], inplace=True)

        return df


    def preprocess_categorical_data(self, df):
    #    sns.countplot(y='occupation', hue='income', data=df)      # show frequency of each category based on 'income'
    #    plt.show()

        replace_map = {'income':{' <=50K':0, ' >50K':1}}
        df.replace(replace_map, inplace=True)

        df = df.apply(preprocessing.LabelEncoder().fit_transform)   # Replace with index after sorting feature

        return df


    def scaler_trainsform(self, X_train, X_test):
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test

