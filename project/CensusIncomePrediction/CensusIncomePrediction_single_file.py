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
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier


ENSEMBLE_TREE_SIZE = 20

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

class UseEnsemble(Enum):
    NoEnsemble = 1
    Bagging = 2
    Boosting = 3


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


def draw_roc_comparison_plot(roc_info, Y_test):
    ns_probs = [0 for _ in range(len(Y_test))]
    ns_fpr, ns_tpr, _ = roc_curve(Y_test, ns_probs)
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')

    for roc_value in roc_info:
        plt.plot(roc_value[1], roc_value[2], label=roc_value[0])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()


def compute_roc_curve(clf, test_X,  test_Y):
    probs = clf.predict_proba(test_X)
    probs = probs[:, 1]

    roc_score = roc_auc_score(test_Y, probs)
    print("ROC-AUC score: ", roc_score)
    fpr, tpr, _ = roc_curve(test_Y, probs)

    return roc_score, [fpr, tpr]


def classify_dataset(estimator, train_X, train_Y, test_X, test_Y, ensemble):
    if (ensemble == UseEnsemble.NoEnsemble):
        clf = estimator
    elif (ensemble == UseEnsemble.Bagging):
        clf = BaggingClassifier(base_estimator=estimator, n_estimators=ENSEMBLE_TREE_SIZE, max_samples = 0.7)
    elif (ensemble == UseEnsemble.Boosting):
        clf = AdaBoostClassifier(base_estimator=estimator, n_estimators=ENSEMBLE_TREE_SIZE, algorithm='SAMME')

    clf.fit(train_X, train_Y)

    print("First 25 prediction: ", clf.predict(test_X[ :25, : ]))
    print("First 25 target:     ", test_Y[:25])

    train_score = clf.score(train_X, train_Y)
    test_score = clf.score(test_X, test_Y)
    print("Training Accuracy: %f" % train_score)
    print("Test Accuracy: %f" % test_score)

    y_true = test_Y
    y_pred = clf.predict(test_X)
    print("Confusion matrix: ")
    cm_score = confusion_matrix(y_true, y_pred)
    print(cm_score)

    roc_score, plot_info = compute_roc_curve(clf, test_X,  test_Y)

    return [train_score, test_score, cm_score[0][0], cm_score[0][1], cm_score[1][0], cm_score[1][1], roc_score], plot_info


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
    wrongly_classified = []

    for i in range(len(predictions)):
        if(predictions[i] ^ Y[i]):
            wrongly_classified.append(i)
        else:
            correctly_classified.append(i)

    X_correct = np.delete(X, wrongly_classified, axis=0)
    Y_correct = np.delete(Y, wrongly_classified, axis=0)
    X_wrong = np.delete(X, correctly_classified, axis=0)
    Y_wrong = np.delete(Y, correctly_classified, axis=0)

    print("shape, X_correct: ", X_correct.shape)
    print("shape, X_wrong: ", X_wrong.shape)

    return X_correct, Y_correct, X_wrong, Y_wrong


def work_with_strong_pattern(clf_strong, name, X_train_correct, Y_train_correct, X_test_correct, Y_test_correct):
    clf_strong, name = get_classifier(ClassifierName.NeuralNetwork)
    clf_strong.fit(X_train_correct, Y_train_correct)

    train_score = clf_strong.score(X_train_correct, Y_train_correct)
    test_score = clf_strong.score(X_test_correct, Y_test_correct)
    print("Training Accuracy: %f" % train_score)
    print("Test Accuracy: %f" % test_score)

    y_pred = clf_strong.predict(X_test_correct)
    print("Confusion matrix: ")
    cm_score = confusion_matrix(Y_test_correct, y_pred)
    print(cm_score)

    _, roc = compute_roc_curve(clf_strong, X_test_correct,  Y_test_correct)
    roc.insert(0, name)
    roc_info = [roc]
    draw_roc_comparison_plot(roc_info, Y_test)


def work_with_weak_pattern(X_train_wrong, Y_train_wrong, X_test_wrong, Y_test_wrong):
    clf_weak, name = get_classifier(ClassifierName.NeuralNetwork)
    _, roc = classify_dataset(clf_weak, X_train_wrong, Y_train_wrong, X_test_wrong, Y_test_wrong, UseEnsemble.NoEnsemble)
    roc.insert(0, name)
    roc_info = [roc]
    draw_roc_comparison_plot(roc_info, Y_test)


def weak_pattern_proof(X_train, X_test, Y_train, Y_test):
    clf, name = get_classifier(ClassifierName.NeuralNetwork)
    _, roc = classify_dataset(clf, X_train, Y_train, X_test, Y_test, UseEnsemble.NoEnsemble)

    roc.insert(0, name)
    roc_info = [roc]
    draw_roc_comparison_plot(roc_info, Y_test)

    train_predictions = clf.predict(X_train)
    test_predictions = clf.predict(X_test)
    X_train_correct, Y_train_correct, X_train_wrong, Y_train_wrong = separate_wrong_predicted_examples(X_train, Y_train, train_predictions)
    X_test_correct, Y_test_correct, X_test_wrong, Y_test_wrong = separate_wrong_predicted_examples(X_test, Y_test, test_predictions)

    work_with_strong_pattern(clf, name, X_train_correct, Y_train_correct, X_test_correct, Y_test_correct)
    work_with_weak_pattern(X_train_wrong, Y_train_wrong, X_test_wrong, Y_test_wrong)


if __name__ == '__main__':
    X_train, Y_train = get_data(DataType.Train, MissingData.ReplaceWithMostFrequentData)
    X_test, Y_test = get_data(DataType.Test, MissingData.ReplaceWithMostFrequentData)

    Y_train = Y_train.values
    Y_test = Y_test.values
    print(X_train.shape)
    print(X_test.shape)

    X_train, X_test = scaler_trainsform(X_train, X_test)       # Mandatory for SVM; Works better for Neural Network

    roc_info = []
    comparison_table = PrettyTable(['Classifier', 'TrainAccuracy', 'TestAccuracy', 'TrueNegative(00) <=50K',
                                    'FalsePositive(01)', 'FalseNegative(10)', 'TruePositive(11) >50K', 'ROC_Score'])

    for clf_name in ClassifierName:
        clf, name = get_classifier(clf_name)
        scores, roc_value = classify_dataset(clf, X_train, Y_train, X_test, Y_test, UseEnsemble.NoEnsemble)  # Boosting != KNeighborsClassifier, MLPClassifier; SVM takes time

        scores.insert(0, name)
        comparison_table.add_row(scores)

        roc_value.insert(0, name)
        roc_info.append(roc_value)

    print("Comparison among all classifiers:")
    print(comparison_table)
    draw_roc_comparison_plot(roc_info, Y_test)

    print("\nWeak pattern proof:")
    weak_pattern_proof(X_train, X_test, Y_train, Y_test)
