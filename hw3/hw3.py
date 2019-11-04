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

    print("Bagging d_tree error: ", run_bagging_model(d_tree))
    print("Boosting d_tree error: ", run_boosting_model(d_tree))

    print("Bagging SVM error: ", run_bagging_model(svm))
    print("Boosting SVM error: ", run_boosting_model(svm))

    print("Bagging SGD error: ", run_bagging_model(sgd))
    print("Boosting SGD error: ", run_boosting_model(sgd))


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
    arr_val = dataset.values
    data = arr_val[:, 0:34]
    target = arr_val[:, 34]
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.4, random_state=0)

    num_trees = 100

#    run_model_only()
    run_model_using_kfold()

