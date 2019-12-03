import numpy as np
import pandas as pd
from sklearn import preprocessing
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
from enum import Enum


class DataType(Enum):
   Train = 1
   Test = 2

class MissingData(Enum):
    Remove = 1                       # Not good approach
    ReplaceWithMostFrequentData = 2  # Better


def preprocess_missing_value(df, processing_type):
    df.replace(' ?', np.NaN, inplace=True)           # replace ' ?' with standard np.nan

#    print(df.isnull().sum())                         # Count missing values per attribute
#    msno.matrix(df, figsize=(10, 6), fontsize=7)     # Plot missing data
#    plt.show()

    if (processing_type == MissingData.Remove):
        print("Remove all examples with missing value")
        df.dropna(inplace=True)
    if (processing_type == MissingData.ReplaceWithMostFrequentData):
        print("Replacing all missing value with most frequent value")
        df.fillna(df.mode().iloc[0], inplace=True)

    return df


def preprocess_categorical_data(df):
#    sns.countplot(y='occupation', hue='income', data=df)      # show frequency of each category based on 'income'
#    plt.show()

    replace_map = {'income':{' >50K':1, ' <=50K':0}}
    df.replace(replace_map, inplace=True)

    df = df.apply(preprocessing.LabelEncoder().fit_transform)   # Replace with index after sorting feature

    return df


def get_data(data_type):
    train_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    test_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

    feature_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

    if data_type == DataType.Train:
        df = pd.read_csv(train_url, delimiter=',', names=feature_names)
    else:
        df = pd.read_csv(test_url, delimiter=',', names=feature_names, skiprows=1)    # First row irrelevant

#    print(df.shape)
#    df.info()

    df = preprocess_missing_value(df, MissingData.ReplaceWithMostFrequentData)        # replace with most frequent data (better)
    df = preprocess_categorical_data(df)

    data = df.iloc[:, 0:14]
    target = df.iloc[:, 14]
    return data, target


if __name__ == '__main__':
    train_X, train_Y = get_data(DataType.Train)
    test_X, test_Y = get_data(DataType.Test)

    print(train_X.shape)
    print(test_X.shape)
