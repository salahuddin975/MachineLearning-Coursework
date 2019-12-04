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
from sklearn.naive_bayes import GaussianNB



class DataType(Enum):
   Train = 1
   Test = 2

class MissingData(Enum):
    RemoveEntireExample = 1               # Not very good approach
    ReplaceWithMostFrequentData = 2       # Better

class ClassifierName(Enum):
    NaiveBayes = 1
    LogisticRegression = 2
    KNeighborsClassifier = 3
    SVM = 4
    DecisionTreeClassifier = 5
    NeuralNetwork = 6


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


def compute_roc_curve(clf, test_X,  Y_test):
    ns_probs = [0 for _ in range(len(Y_test))]
    ns_fpr, ns_tpr, _ = roc_curve(Y_test, ns_probs)
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')

    probs = clf.predict_proba(test_X)
    probs = probs[:, 1]

    roc_score = roc_auc_score(Y_test, probs)
    print("ROC-AUC score: ", roc_score)
    fpr, tpr, _ = roc_curve(Y_test, probs)
    plt.plot(fpr, tpr, label="Classifier")

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()


def classify_dataset(clf, train_X, train_Y, test_X, test_Y):
    clf.fit(train_X, train_Y)

    train_predictions = clf.predict(train_X)
    test_predictions = clf.predict(test_X)
    print("First 25 prediction: ", train_predictions[:25])
    print("First 25 target:     ", train_Y[:25])

    train_score = clf.score(train_X, train_Y)
    test_score = clf.score(test_X, test_Y)
    print("Training Accuracy: %f" % train_score)
    print("Test Accuracy: %f" % test_score)

    y_pred = clf.predict(test_X)
    print("Confusion matrix: ")
    cm_score = confusion_matrix(test_Y, y_pred)
    print(cm_score)

    compute_roc_curve(clf, test_X,  test_Y)

    return train_predictions, test_predictions


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
        clf = MLPClassifier(solver='adam', hidden_layer_sizes = (5, 2), max_iter=500, alpha=1e-5, random_state = 1)  # (5, 2) works better
    elif (clf_name == ClassifierName.NaiveBayes):
        print("\nUsing classifier: GaussianNB()")
        name = "NaiveBayes"
        clf = GaussianNB()

    return clf, name


def separate_wrong_predicted_examples(X, Y, predictions):
    correctly_classified = []
    Y = Y.values

    for i in range(len(predictions)):
        if(predictions[i] ^ Y[i] == 0):
            correctly_classified.append(i)

    print("correctly classified: ", len(correctly_classified))

    X = np.delete(X, correctly_classified, axis=0)
    Y = np.delete(Y, correctly_classified, axis=0)
#    Y.drop(Y.index[correctly_classified], inplace=True)
    print("shape, X_train: ", X.shape)
    print("shape, Y_train: ", Y.shape)

    return X, Y


if __name__ == '__main__':
    X_train, Y_train = get_data(DataType.Train, MissingData.ReplaceWithMostFrequentData)
    X_test, Y_test = get_data(DataType.Test, MissingData.ReplaceWithMostFrequentData)

    print(X_train.shape)
    print(X_test.shape)

    X_train, X_test = scaler_trainsform(X_train, X_test)       # Mandatory for SVM; Works better for Neural Network

    clf, name = get_classifier(ClassifierName.NaiveBayes)
    train_predictions, test_predictions = classify_dataset(clf, X_train, Y_train, X_test, Y_test)

    X_train_wrong, Y_train_wrong = separate_wrong_predicted_examples(X_train, Y_train, train_predictions)
    X_test_wrong, Y_test_wrong = separate_wrong_predicted_examples(X_test, Y_test, test_predictions)

    clf_wrong, name = get_classifier(ClassifierName.NeuralNetwork)
    predictions = classify_dataset(clf_wrong, X_train_wrong, Y_train_wrong, X_test_wrong, Y_test_wrong)

