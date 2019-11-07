import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np



def plot_comparison_without_k_fold(d_tree_errors, svm_errors, sgd_errors):
    n_groups = 2

    plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.25
    opacity = 0.8

    plt.bar(index, d_tree_errors, bar_width, alpha=opacity, color='b', label='D-Tree')
    plt.bar(index + bar_width, svm_errors, bar_width, alpha=opacity, color='g', label='SVM')
    plt.bar(index + 2*bar_width, sgd_errors, bar_width, alpha=opacity, color='r', label='SGD')

    plt.xlabel('Estimator')
    plt.ylabel('Error')
    plt.title('Comparison among different estimators')
    plt.xticks(index + bar_width, ('Bagging', 'Boosting'))
    plt.legend()

    plt.tight_layout()
    plt.show()


def run_bagging_model(estimator):
    clf = BaggingClassifier(base_estimator=estimator, n_estimators=num_trees, max_samples = 0.6)
    clf.fit(x_train, y_train)
    error = 1 - clf.score(x_test, y_test)
    return error


def run_boosting_model(estimator):
    clf = AdaBoostClassifier(base_estimator=estimator, n_estimators=num_trees, algorithm='SAMME')
    clf.fit(x_train, y_train)
    error = 1 - clf.score(x_test, y_test)
    return error


def run_model_only():
    d_tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=1)
    svm = SVC(probability=True, kernel='linear')
    sgd = SGDClassifier(loss='hinge')

    d_tree_errors = []
    d_tree_errors.append(run_bagging_model(d_tree))
    d_tree_errors.append(run_boosting_model(d_tree))

    svm_errors = []
    svm_errors.append(run_bagging_model(svm))
    svm_errors.append(run_boosting_model(svm))

    sgd_errors = []
    sgd_errors.append(run_bagging_model(sgd))
    sgd_errors.append(run_boosting_model(sgd))

    plot_comparison_without_k_fold(d_tree_errors, svm_errors, sgd_errors)


def run_bagging_with_cross_validation(estimator, k_fold):
    kfold = KFold(n_splits=k_fold)
    clf = BaggingClassifier(base_estimator=estimator, n_estimators=num_trees)
    scores = cross_val_score(clf, x_train, y_train, cv=kfold)
    # Accuracy should come from test
    return 1 - scores.mean()


def run_boosting_with_cross_validation(estimator, k_fold):
    kfold = KFold(n_splits=k_fold)
    clf = AdaBoostClassifier(base_estimator=estimator, n_estimators=num_trees, algorithm='SAMME')
    scores = cross_val_score(clf, x_train, y_train, cv=kfold)
    # Accuracy should come from test
    return 1 - scores.mean()


def run_model_using_kfold():
    d_tree = DecisionTreeClassifier()
    svm = SVC(probability=True, kernel='linear')
    sgd = SGDClassifier(loss='hinge')

    k_fold = 0
    for i in range(5):
        k_fold = k_fold + 2
        print("k_fold: ", k_fold, ", Bagging d_tree error: ", run_bagging_with_cross_validation(d_tree, k_fold))
        print("k_fold: ", k_fold, ", Boosting d_tree error: ", run_boosting_with_cross_validation(d_tree, k_fold))


if __name__ == '__main__':
    dataset = pd.read_csv('ionosphere.csv')
#    print(dataset)

    arr_val = dataset.values
    data = arr_val[:, 0:34]
    target = arr_val[:, 34]
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=0)

    num_trees = 100

    run_model_only()
#    run_model_using_kfold()

