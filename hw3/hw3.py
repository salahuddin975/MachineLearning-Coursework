import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve


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
    plt.xlabel("K-fold")
    plt.ylabel("Error rate")
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
    plt.xlabel("K-fold")
    plt.ylabel("Error rate")
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
    plt.xlabel("K-fold")
    plt.ylabel("Error rate")
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
    plt.xlabel("K-fold")
    plt.ylabel("Error rate")
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
    plt.xlabel("K-fold")
    plt.ylabel("Error rate")
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
    plt.xlabel("K-fold")
    plt.ylabel("Error rate")
    plt.title("Bagging Boosting comparison using SGD")
    plt.show()


def plot_test_train_error_vs_complexity(train_error, test_error):
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    plt.plot(x, train_error, label="train_error")
    plt.plot(x, test_error, label="test_error")

    plt.legend(loc = 1, fancybox=True, framealpha=0.5)
    plt.xticks(x)
    plt.xlabel("Testing Fold No.")
    plt.ylabel("Error rate")
    plt.title("Graph for D-tree(Bagging)")
    plt.show()


def plot_tree_depth_vs_complexity(bagging_train_errors, bagging_test_errors, boosting_train_errors, boosting_test_errors):
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
#    plt.plot(x, bagging_train_errors, label="bagging train error")
#    plt.plot(x, bagging_test_errors, label="bagging test error")
    plt.plot(x, boosting_train_errors, label="boosting train error")
    plt.plot(x, boosting_test_errors, label="boosting test error")

    plt.legend(loc = 1, fancybox=True, framealpha=0.5)
    plt.xticks(x)
    plt.xlabel("Depth of the tree")
    plt.ylabel("Error rate")
    plt.title("D-tree depth vs train and test error")
    plt.show()


def plot_complexity_number_of_trees(train_error, test_error):
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    plt.plot(x, train_error, label="train_error")
    plt.plot(x, test_error, label="test_error")

    plt.legend(loc = 1, fancybox=True, framealpha=0.5)
    plt.xticks(x)
    plt.xlabel("Number of bags")
    plt.ylabel("Error rate")
    plt.title("D-tree number of bags vs train and test error(Bagging)")
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


def run_bagging_with_cross_validation(estimator, k_fold, num_trees):
    print(num_trees)
    clf = BaggingClassifier(base_estimator=estimator, n_estimators=num_trees)
    scores = cross_validate(clf, ionosphere_data, ionosphere_target, return_train_score = True, cv=k_fold)
    train_error = 1 - scores['train_score'].mean()
    test_error = 1 - scores['test_score'].mean()

    return train_error, test_error


def run_boosting_with_cross_validation(estimator, k_fold, num_trees):
    print(num_trees)
    clf = AdaBoostClassifier(base_estimator=estimator, n_estimators=num_trees, algorithm='SAMME')
    scores = cross_validate(clf, ionosphere_data, ionosphere_target, return_train_score = True, cv=k_fold)
    train_error = 1 - scores['train_score'].mean()
    test_error = 1 - scores['test_score'].mean()

    return train_error, test_error


def run_model_using_kfold():
    num_trees = 50
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
        scores = run_bagging_with_cross_validation(d_tree, k_fold, num_trees)
        bagging_d_tree_train_errors.append(scores[0])
        bagging_d_tree_test_errors.append(scores[1])

        scores = run_bagging_with_cross_validation(svm, k_fold, num_trees)
        bagging_svm_train_errors.append(scores[0])
        bagging_svm_test_errors.append(scores[1])

        scores = run_bagging_with_cross_validation(sgd, k_fold, num_trees)
        bagging_sgd_train_errors.append(scores[0])
        bagging_sgd_test_errors.append(scores[1])

        scores = run_boosting_with_cross_validation(d_tree, k_fold, num_trees)
        boosting_d_tree_train_errors.append(scores[0])
        boosting_d_tree_test_errors.append(scores[1])

        scores = run_boosting_with_cross_validation(svm, k_fold, num_trees)
        boosting_svm_train_errors.append(scores[0])
        boosting_svm_test_errors.append(scores[1])

        scores = run_boosting_with_cross_validation(sgd, k_fold, num_trees)
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


def get_train_and_test_error_bagging(estimator, k_fold):
    clf = BaggingClassifier(base_estimator=estimator, n_estimators=num_trees)
    scores = cross_validate(clf, ionosphere_data, ionosphere_target, return_train_score=True, cv=k_fold)
    train_errors = 1 - scores['train_score']
    test_errors = 1 - scores['test_score']
    return train_errors, test_errors


def get_train_and_test_error_boosting(estimator, k_fold):
    clf = BaggingClassifier(base_estimator=estimator, n_estimators=num_trees)
    scores = cross_validate(clf, ionosphere_data, ionosphere_target, return_train_score=True, cv=k_fold)
    train_errors = 1 - scores['train_score']
    test_errors = 1 - scores['test_score']
    return train_errors, test_errors


def find_complexity():
    k_fold = 8

    d_tree = DecisionTreeClassifier()
    svm = SVC(probability=True, kernel='linear')
    sgd = SGDClassifier(loss='hinge')

    train_errors, test_errors = get_train_and_test_error_bagging(d_tree, k_fold)
    plot_test_train_error_vs_complexity(train_errors, test_errors)

#    train_errors, test_errors = get_train_and_test_error_boosting(svm, k_fold)
#    plot_test_train_error_vs_complexity(train_errors, test_errors)


def find_complexity_with_d_tree_depth():
    k_fold = 8
    num_trees = 20
    bagging_train_errors = []
    bagging_test_errors = []
    boosting_train_errors = []
    boosting_test_errors = []

    for i in range(1, 13):
        print(i)
        d_tree = DecisionTreeClassifier(max_depth=i)
        bagging_scores = run_bagging_with_cross_validation(d_tree, k_fold, num_trees)
        boosting_scores = run_boosting_with_cross_validation(d_tree, k_fold, num_trees)
        print(bagging_scores[0], bagging_scores[1])
        bagging_train_errors.append(bagging_scores[0])
        bagging_test_errors.append(bagging_scores[1])
        boosting_train_errors.append(boosting_scores[0])
        boosting_test_errors.append(boosting_scores[1])

    plot_tree_depth_vs_complexity(bagging_train_errors, bagging_test_errors, boosting_train_errors, boosting_test_errors)



def find_complexity_with_number_of_bags():
    k_fold = 8
    train_errors = []
    test_errors = []
    d_tree = DecisionTreeClassifier(max_depth=6)
    svm = SVC(probability=True, kernel='linear')
    sgd = SGDClassifier(loss='hinge')

    num_trees = 0
    for i in range(15):
        num_trees = num_trees + 1
        scores = run_bagging_with_cross_validation(d_tree, k_fold, num_trees)
#        scores = run_boosting_with_cross_validation(d_tree, k_fold, num_trees)
        print(scores[0], scores[1])
        train_errors.append(scores[0])
        test_errors.append(scores[1])

    plot_complexity_number_of_trees(train_errors, test_errors)


if __name__ == '__main__':
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
    dataset = pd.read_csv(url, delimiter=',')

    arr_val = dataset.values
    ionosphere_data = arr_val[:, 0:34]
    ionosphere_target = arr_val[:, 34]
    x_train, x_test, y_train, y_test = train_test_split(ionosphere_data, ionosphere_target, test_size=0.3, random_state=0)

    num_trees = 50

    run_model_only()
    run_model_using_kfold()
    find_complexity()
    find_complexity_with_d_tree_depth()
    find_complexity_with_number_of_bags()
