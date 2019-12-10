import numpy as np
from Constants import UseEnsemble
from Constants import ClassifierName
from DrawPlots import DrawPlot
from ClassifyDataset import DatasetClassify
from DatasetPreprocessing import DataPreprocessing
from sklearn.metrics import confusion_matrix


class WeakPattern:
    def __init__(self):
        pass

    def separate_wrong_predicted_examples(self, X, Y, predictions):
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


    def work_with_strong_pattern(self, clf_strong, name, X_train_correct, Y_train_correct, X_test_correct, Y_test_correct):
        dc = DatasetClassify()
        clf_strong, name = dc.get_classifier(ClassifierName.NeuralNetwork)
        clf_strong.fit(X_train_correct, Y_train_correct)

        train_score = clf_strong.score(X_train_correct, Y_train_correct)
        test_score = clf_strong.score(X_test_correct, Y_test_correct)
        print("Training Accuracy: %f" % train_score)
        print("Test Accuracy: %f" % test_score)

        dp = DrawPlot()
        y_pred = clf_strong.predict(X_test_correct)
        print("Confusion matrix: ")
        cm_score = confusion_matrix(Y_test_correct, y_pred)
        print(cm_score)

        _, roc = dp.compute_roc_curve(clf_strong, X_test_correct,  Y_test_correct)
        roc.insert(0, name)
        roc_info = [roc]
        dp.draw_roc_comparison_plot(roc_info, Y_test_correct)


    def work_with_weak_pattern(self, X_train_wrong, Y_train_wrong, X_test_wrong, Y_test_wrong):
        dp = DrawPlot()
        dc = DatasetClassify()
        clf_weak, name = dc.get_classifier(ClassifierName.NeuralNetwork)
        _, roc = dc.classify_dataset(clf_weak, X_train_wrong, Y_train_wrong, X_test_wrong, Y_test_wrong, UseEnsemble.NoEnsemble)
        roc.insert(0, name)
        roc_info = [roc]
        dp.draw_roc_comparison_plot(roc_info, Y_test_wrong)


    def weak_pattern_proof(self, X_train, X_test, Y_train, Y_test):
        dp = DrawPlot()
        dc = DatasetClassify()

        clf, name = dc.get_classifier(ClassifierName.NeuralNetwork)
        _, roc = dc.classify_dataset(clf, X_train, Y_train, X_test, Y_test, UseEnsemble.NoEnsemble)

        roc.insert(0, name)
        roc_info = [roc]
        dp.draw_roc_comparison_plot(roc_info, Y_test)

        train_predictions = clf.predict(X_train)
        test_predictions = clf.predict(X_test)
        X_train_correct, Y_train_correct, X_train_wrong, Y_train_wrong = self.separate_wrong_predicted_examples(X_train, Y_train, train_predictions)
        X_test_correct, Y_test_correct, X_test_wrong, Y_test_wrong = self.separate_wrong_predicted_examples(X_test, Y_test, test_predictions)

        self.work_with_strong_pattern(clf, name, X_train_correct, Y_train_correct, X_test_correct, Y_test_correct)
        self.work_with_weak_pattern(X_train_wrong, Y_train_wrong, X_test_wrong, Y_test_wrong)
