from DrawPlots import DrawPlot
from Constants import UseEnsemble
from Constants import ClassifierName
from Constants import ENSEMBLE_TREE_SIZE
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier


class DatasetClassify:
    def __init__(self):
        pass

    def classify_dataset(self, estimator, train_X, train_Y, test_X, test_Y, ensemble):
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

        dp = DrawPlot()
        roc_score, plot_info = dp.compute_roc_curve(clf, test_X,  test_Y)

        return [train_score, test_score, cm_score[0][0], cm_score[0][1], cm_score[1][0], cm_score[1][1], roc_score], plot_info


    def get_classifier(self, clf_name):
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


