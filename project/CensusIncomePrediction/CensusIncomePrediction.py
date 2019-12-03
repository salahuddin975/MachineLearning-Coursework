import numpy as np
import pandas as pd
from enum import Enum
import seaborn as sns
import missingno as msno
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


class DataType(Enum):
   Train = 1
   Test = 2

class MissingData(Enum):
    RemoveEntireExample = 1                       # Not good approach
    ReplaceWithMostFrequentData = 2  # Better

class ClassifierName(Enum):
    LogisticRegression = 1
    KNeighborsClassifier = 2
    DecisionTreeClassifier = 3
    NeuralNetwork = 4


def preprocess_missing_value(df, processing_type):
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


def preprocess_categorical_data(df):
#    sns.countplot(y='occupation', hue='income', data=df)      # show frequency of each category based on 'income'
#    plt.show()

    replace_map = {'income':{' <=50K':0, ' >50K':1}}
    df.replace(replace_map, inplace=True)

    df = df.apply(preprocessing.LabelEncoder().fit_transform)   # Replace with index after sorting feature

    return df


def get_data(data_type, action_for_missing_value):
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

    df = preprocess_missing_value(df, action_for_missing_value)
    df = preprocess_categorical_data(df)

    data = df.iloc[:, 0:14]
    target = df.iloc[:, 14]
    return data, target


def compute_roc_curve(clf, test_X,  test_Y):
    ns_probs = [0 for _ in range(len(test_Y))]
    lr_probs = clf.predict_proba(test_X)
    lr_probs = lr_probs[:, 1]

    lr_auc = roc_auc_score(test_Y, lr_probs)
    print("\nROC-AUC score: ", lr_auc)

    ns_fpr, ns_tpr, _ = roc_curve(test_Y, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(test_Y, lr_probs)

    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Classifier')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()


def classify_dataset(clf, train_X, train_Y, test_X, test_Y):
    clf.fit(train_X, train_Y)

    print("First 25 prediction: ", clf.predict(test_X.iloc[ :25, : ]))
    print("First 25 target:     ", test_Y[:25].values)

    print("\nTraining Accuracy: %f" % clf.score(train_X, train_Y))
    print("Test Accuracy: %f" % clf.score(test_X, test_Y))

    y_true = test_Y.values
    y_pred = clf.predict(test_X)
    print("\nConfusion matrix: ")
    print("00(true negative, <=50K)   01(false positive)")
    print("10(false negative)         11(true positive, >50K)")
    print(confusion_matrix(y_true, y_pred))

    compute_roc_curve(clf, test_X,  test_Y)


def get_classifier(clf_name):
    clf = None

    if (clf_name == ClassifierName.KNeighborsClassifier):
        print("Using classifier: KNeighborsClassifier")
        clf = KNeighborsClassifier()
    elif (clf_name == ClassifierName.LogisticRegression):
        print("Using classifier: LogisticRegression (solver='lbfgs', max_iter=400)")
        clf = LogisticRegression(solver='lbfgs', max_iter=400)
    elif (clf_name == ClassifierName.DecisionTreeClassifier):
        print("Using classifier: DecisionTreeClassifier (max_depth = 12)")
        clf = DecisionTreeClassifier(max_depth=12)
    elif (clf_name == ClassifierName.NeuralNetwork):
        print("Using classifier: NeuralNetwork (solver='adam', hidden_layer_sizes = (5, 2))")
        clf = MLPClassifier(solver='adam', hidden_layer_sizes = (5, 2), alpha=1e-5, random_state = 1, verbose=True)

#    mlp = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=150, solver='adam', verbose=True, learning_rate_init=.1)
#    svm = SVC(probability=True, kernel='linear')

    return clf


if __name__ == '__main__':
    train_X, train_Y = get_data(DataType.Train, MissingData.ReplaceWithMostFrequentData)
    test_X, test_Y = get_data(DataType.Test, MissingData.ReplaceWithMostFrequentData)

    print(train_X.shape)
    print(test_X.shape)

    clf = get_classifier(ClassifierName.NeuralNetwork)
    classify_dataset(clf, train_X, train_Y, test_X, test_Y)

