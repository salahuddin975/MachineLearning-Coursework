import numpy as np
import pandas as pd
from enum import Enum
import seaborn as sns
import missingno as msno
from prettytable import PrettyTable
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
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
    SVM = 3
    DecisionTreeClassifier = 4
    NeuralNetwork = 5


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


def scaler_trainsform(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test


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

    roc_score = roc_auc_score(test_Y, lr_probs)
    print("\nROC-AUC score: ", roc_score)

    ns_fpr, ns_tpr, _ = roc_curve(test_Y, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(test_Y, lr_probs)

    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Classifier')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
#    plt.show()

    return roc_score


def classify_dataset(clf, train_X, train_Y, test_X, test_Y):
    clf.fit(train_X, train_Y)

    print("First 25 prediction: ", clf.predict(test_X[ :25, : ]))
    print("First 25 target:     ", test_Y[:25].values)

    train_score = clf.score(train_X, train_Y)
    test_score = clf.score(test_X, test_Y)
    print("\nTraining Accuracy: %f" % train_score)
    print("Test Accuracy: %f" % test_score)

    y_true = test_Y.values
    y_pred = clf.predict(test_X)
    print("\nConfusion matrix: ")
    cm_score = confusion_matrix(y_true, y_pred)
    print(cm_score)

    roc_score = compute_roc_curve(clf, test_X,  test_Y)

    return [train_score, test_score, cm_score[0][0], cm_score[0][1], cm_score[1][0], cm_score[1][1], roc_score]


def get_classifier(clf_name):
    clf = None
    name = ""

    if (clf_name == ClassifierName.KNeighborsClassifier):
        print("\nUsing classifier: KNeighborsClassifier")
        name = "KNeighborsClassifier"
        clf = KNeighborsClassifier()
    elif (clf_name == ClassifierName.LogisticRegression):
        print("\nUsing classifier: LogisticRegression (solver='lbfgs', max_iter=400)")
        name = "LogisticRegression"
        clf = LogisticRegression(solver='lbfgs', max_iter=400)
    elif (clf_name == ClassifierName.SVM):
        print("\nUsing classifier: SVM (kernel='rbf', probability=True, gamma='scale')")
        name = "SVM"
        clf = svm.SVC(kernel='rbf', probability=True, gamma='scale')
    elif (clf_name == ClassifierName.DecisionTreeClassifier):
        print("\nUsing classifier: DecisionTreeClassifier (max_depth = 12)")
        name = "DecisionTree"
        clf = DecisionTreeClassifier(max_depth=12)
    elif (clf_name == ClassifierName.NeuralNetwork):
        print("\nUsing classifier: NeuralNetwork (solver='adam', hidden_layer_sizes = (5, 2))")
        name = "NeuralNetwork"
        clf = MLPClassifier(solver='adam', hidden_layer_sizes = (5, 2), alpha=1e-5, random_state = 1)  # (5, 2) works better

    return clf, name


if __name__ == '__main__':
    X_train, Y_train = get_data(DataType.Train, MissingData.ReplaceWithMostFrequentData)
    X_test, Y_test = get_data(DataType.Test, MissingData.ReplaceWithMostFrequentData)

    print(X_train.shape)
    print(X_test.shape)

    X_train, X_test = scaler_trainsform(X_train, X_test)  # Mandatory for SVM; Works better for Neural Network
    comparison_table = PrettyTable(['Classifier', 'TrainAccuracy', 'TestAccuracy',
                                    'TrueNegative(00) <=50K', 'FalsePositive(01)', 'FalseNegative(10)', 'TruePositive(11) >50K', 'ROC_Score'])

    for clf_name in ClassifierName:
        clf, name = get_classifier(clf_name)
        scores = classify_dataset(clf, X_train, Y_train, X_test, Y_test)
        scores.insert(0, name)
        comparison_table.add_row(scores)

    print(comparison_table)
