import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score


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


def test_train_error_vs_k_fold(bagging_d_tree_train_errors, bagging_svm_train_errors, bagging_sgd_train_errors,
                               bagging_d_tree_test_errors, bagging_svm_test_errors, bagging_sgd_test_errors,
                               boosting_d_tree_train_errors, boosting_svm_train_errors, boosting_sgd_train_errors,
                               boosting_d_tree_test_errors, boosting_svm_test_errors, boosting_sgd_test_errors ):

    x = [2, 4, 6, 8, 10]
    fig, ax = plt.subplots()

    ax.plot(x, bagging_d_tree_train_errors, label="Bagging d-tree train errors")
    ax.plot(x, bagging_d_tree_test_errors, label="Bagging d-tree test errors")
    ax.plot(x, bagging_svm_train_errors, label="Bagging svm train errors")
    ax.plot(x, bagging_svm_test_errors, label="Bagging svm test errors")
    ax.plot(x, bagging_sgd_train_errors, label="Bagging sgd train errors")
    ax.plot(x, bagging_sgd_test_errors, label="Bagging sgd test errors")

    ax.plot(x, boosting_d_tree_train_errors, label="Boosting d-tree train errors")
    ax.plot(x, boosting_d_tree_test_errors, label="Boosting d-tree test errors")
    ax.plot(x, boosting_svm_train_errors, label="Boosting svm train errors")
    ax.plot(x, boosting_svm_test_errors, label="Boosting svm test errors")
    ax.plot(x, boosting_sgd_train_errors, label="Boosting sgd train errors")
    ax.plot(x, boosting_sgd_test_errors, label="Boosting sgd test errors")
    ax.legend(loc = 4, fancybox=True, framealpha=0.5)

    plt.title("Comparisons among different estimators for Bagging and Boosting")
    plt.xticks(x)
    plt.show()


def bagging_test_train_error_vs_k_fold(bagging_d_tree_train_errors, bagging_svm_train_errors, bagging_sgd_train_errors,
                               bagging_d_tree_test_errors, bagging_svm_test_errors, bagging_sgd_test_errors):
    x = [2, 4, 6, 8, 10]

    fig, ax = plt.subplots()
    ax.plot(x, bagging_d_tree_train_errors, label="d-tree train errors")
    ax.plot(x, bagging_d_tree_test_errors, label="d-tree test errors")
    ax.plot(x, bagging_svm_train_errors, label="svm train errors")
    ax.plot(x, bagging_svm_test_errors, label="svm test errors")
    ax.plot(x, bagging_sgd_train_errors, label="sgd train errors")
    ax.plot(x, bagging_sgd_test_errors, label="sgd test errors")
    ax.legend(loc = 4, fancybox=True, framealpha=0.5)

    plt.title("Comparisons among different estimators using Bagging")
    plt.xticks(x)
    plt.show()


def boosting_test_train_error_vs_k_fold(boosting_d_tree_train_errors, boosting_svm_train_errors, boosting_sgd_train_errors,
                               boosting_d_tree_test_errors, boosting_svm_test_errors, boosting_sgd_test_errors ):

    x = [2, 4, 6, 8, 10]
    fig, ax = plt.subplots()

    ax.plot(x, boosting_d_tree_train_errors, label="d-tree train errors")
    ax.plot(x, boosting_d_tree_test_errors, label="d-tree test errors")
    ax.plot(x, boosting_svm_train_errors, label="svm train errors")
    ax.plot(x, boosting_svm_test_errors, label="svm test errors")
    ax.plot(x, boosting_sgd_train_errors, label="sgd train errors")
    ax.plot(x, boosting_sgd_test_errors, label="sgd test errors")

    ax.legend(loc = 4, fancybox=True, framealpha=0.5)
    plt.xticks(x)
    plt.title("Comparisons among different estimators using Boosting")
    plt.show()


def d_tree_test_train_error_vs_k_fold(bagging_d_tree_train_errors, bagging_d_tree_test_errors,
                                      boosting_d_tree_train_errors, boosting_d_tree_test_errors):
    x = [2, 4, 6, 8, 10]

    plt.plot(x, bagging_d_tree_train_errors,  label="Bagging d-tree train errors")
    plt.plot(x, bagging_d_tree_test_errors, label="Bagging d-tree test errors")
    plt.plot(x, boosting_d_tree_train_errors, label="Boosting d-tree train errors")
    plt.plot(x, boosting_d_tree_test_errors, label="Boosting d-tree test errors")

    plt.title("Bagging Boosting comparison using d-tree")
    plt.xticks(x)
    plt.legend(loc = 4, fancybox=True, framealpha=0.5)
    plt.show()


def svm_test_train_error_vs_k_fold(bagging_svm_train_errors, bagging_svm_test_errors,
                                   boosting_svm_train_errors, boosting_svm_test_errors):


    x = [2, 4, 6, 8, 10]
    fig, ax = plt.subplots()
    ax.plot(x, bagging_svm_train_errors, label="Bagging svm train errors")
    ax.plot(x, bagging_svm_test_errors, label="Bagging svm test errors")
    ax.plot(x, boosting_svm_train_errors, label="Boosting svm train errors")
    ax.plot(x, boosting_svm_test_errors, label="Boosting svm test errors")
    ax.legend(loc = 4, fancybox=True, framealpha=0.5)

    plt.title("Bagging Boosting comparison using SVM")
    plt.xticks(x)
    plt.show()


def sgd_test_train_error_vs_k_fold(bagging_sgd_train_errors, bagging_sgd_test_errors,
                                   boosting_sgd_train_errors, boosting_sgd_test_errors ):

    x = [2, 4, 6, 8, 10]
    fig, ax = plt.subplots()

    ax.plot(x, bagging_sgd_train_errors, label="Bagging sgd train errors")
    ax.plot(x, bagging_sgd_test_errors, label="Bagging sgd test errors")
    ax.plot(x, boosting_sgd_train_errors, label="Boosting sgd train errors")
    ax.plot(x, boosting_sgd_test_errors, label="Boosting sgd test errors")

    ax.legend(loc = 4, fancybox=True, framealpha=0.5)
    plt.xticks(x)
    plt.title("Bagging Boosting comparison using SGD")
    plt.show()


def run_bagging_model(estimator):
    clf = BaggingClassifier(base_estimator=estimator, n_estimators=num_trees, max_samples = 0.7)
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
    clf = BaggingClassifier(base_estimator=estimator, n_estimators=num_trees)
    scores = cross_validate(clf, ionosphere_data, ionosphere_target, return_train_score = True, cv=k_fold)
    train_error = 1 - scores['train_score'].mean()
    test_error = 1 - scores['test_score'].mean()

    return train_error, test_error


def run_boosting_with_cross_validation(estimator, k_fold):
    clf = AdaBoostClassifier(base_estimator=estimator, n_estimators=num_trees, algorithm='SAMME')
    scores = cross_validate(clf, ionosphere_data, ionosphere_target, return_train_score = True, cv=k_fold)
    train_error = 1 - scores['train_score'].mean()
    test_error = 1 - scores['test_score'].mean()

    return train_error, test_error


def run_model_using_kfold():
    d_tree = DecisionTreeClassifier()
    svm = SVC(probability=True, kernel='linear')
    sgd = SGDClassifier(loss='hinge')

    k_fold = 0
    bagging_d_tree_train_errors = []
    bagging_svm_train_errors = []
    bagging_sgd_train_errors = []
    bagging_d_tree_test_errors = []
    bagging_svm_test_errors = []
    bagging_sgd_test_errors = []

    boosting_d_tree_train_errors = []
    boosting_svm_train_errors = []
    boosting_sgd_train_errors = []
    boosting_d_tree_test_errors = []
    boosting_svm_test_errors = []
    boosting_sgd_test_errors = []

    for i in range(5):
        k_fold = k_fold + 2
        scores = run_bagging_with_cross_validation(d_tree, k_fold)
        bagging_d_tree_train_errors.append(scores[0])
        bagging_d_tree_test_errors.append(scores[1])

        scores = run_bagging_with_cross_validation(svm, k_fold)
        bagging_svm_train_errors.append(scores[0])
        bagging_svm_test_errors.append(scores[1])

        scores = run_bagging_with_cross_validation(sgd, k_fold)
        bagging_sgd_train_errors.append(scores[0])
        bagging_sgd_test_errors.append(scores[1])

        scores = run_boosting_with_cross_validation(d_tree, k_fold)
        boosting_d_tree_train_errors.append(scores[0])
        boosting_d_tree_test_errors.append(scores[1])

        scores = run_boosting_with_cross_validation(svm, k_fold)
        boosting_svm_train_errors.append(scores[0])
        boosting_svm_test_errors.append(scores[1])

        scores = run_boosting_with_cross_validation(sgd, k_fold)
        boosting_sgd_train_errors.append(scores[0])
        boosting_sgd_test_errors.append(scores[1])

    test_train_error_vs_k_fold(bagging_d_tree_train_errors, bagging_svm_train_errors, bagging_sgd_train_errors,
                               bagging_d_tree_test_errors, bagging_svm_test_errors, bagging_sgd_test_errors,
                               boosting_d_tree_train_errors, boosting_svm_train_errors, boosting_sgd_train_errors,
                               boosting_d_tree_test_errors, boosting_svm_test_errors, boosting_sgd_test_errors )

    bagging_test_train_error_vs_k_fold(bagging_d_tree_train_errors, bagging_svm_train_errors, bagging_sgd_train_errors,
                                           bagging_d_tree_test_errors, bagging_svm_test_errors, bagging_sgd_test_errors)

    boosting_test_train_error_vs_k_fold(boosting_d_tree_train_errors, boosting_svm_train_errors, boosting_sgd_train_errors,
                               boosting_d_tree_test_errors, boosting_svm_test_errors, boosting_sgd_test_errors )

    d_tree_test_train_error_vs_k_fold(bagging_d_tree_train_errors, bagging_d_tree_test_errors,
                                      boosting_d_tree_train_errors, boosting_d_tree_test_errors)

    svm_test_train_error_vs_k_fold(bagging_svm_train_errors, bagging_svm_test_errors,
                                       boosting_svm_train_errors, boosting_svm_test_errors)

    sgd_test_train_error_vs_k_fold(bagging_sgd_train_errors, bagging_sgd_test_errors,
                                   boosting_sgd_train_errors, boosting_sgd_test_errors )


if __name__ == '__main__':
    dataset = pd.read_csv('ionosphere.csv')
#    print(dataset)

    arr_val = dataset.values
    ionosphere_data = arr_val[:, 0:34]
    ionosphere_target = arr_val[:, 34]
    x_train, x_test, y_train, y_test = train_test_split(ionosphere_data, ionosphere_target, test_size=0.3, random_state=0)

    num_trees = 100

#    run_model_only()
    run_model_using_kfold()

