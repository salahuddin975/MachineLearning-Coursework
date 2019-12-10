import pandas as pd
from prettytable import PrettyTable
from DatasetPreprocessing import DataPreprocessing
from DrawPlots import DrawPlot
from Constants import DataType
from Constants import MissingData
from Constants import ClassifierName
from Constants import UseEnsemble
from ClassifyDataset import DatasetClassify
from WeakPatternProof import WeakPattern


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

    dp = DataPreprocessing()
    df = dp.preprocess_missing_value(df, action_for_missing_value)
    df = dp.preprocess_categorical_data(df)

    data = df.iloc[:, 0:14]
    target = df.iloc[:, 14]
    return data, target


if __name__ == '__main__':
    X_train, Y_train = get_data(DataType.Train, MissingData.ReplaceWithMostFrequentData)
    X_test, Y_test = get_data(DataType.Test, MissingData.ReplaceWithMostFrequentData)

    Y_train = Y_train.values
    Y_test = Y_test.values
    print(X_train.shape)
    print(X_test.shape)

    dp = DataPreprocessing()
    X_train, X_test = dp.scaler_trainsform(X_train, X_test)       # Mandatory for SVM; Works better for Neural Network

    roc_info = []
    cd = DatasetClassify()
    dp = DrawPlot()
    comparison_table = PrettyTable(['Classifier', 'TrainAccuracy', 'TestAccuracy', 'TrueNegative(00) <=50K',
                                    'FalsePositive(01)', 'FalseNegative(10)', 'TruePositive(11) >50K', 'ROC_Score'])

    for clf_name in ClassifierName:
        clf, name = cd.get_classifier(clf_name)
        scores, roc_value = cd.classify_dataset(clf, X_train, Y_train, X_test, Y_test, UseEnsemble.NoEnsemble)  # Boosting != KNeighborsClassifier, MLPClassifier; SVM takes time

        scores.insert(0, name)
        comparison_table.add_row(scores)

        roc_value.insert(0, name)
        roc_info.append(roc_value)

    print("Comparison among all classifiers:")
    print(comparison_table)
    dp.draw_roc_comparison_plot(roc_info, Y_test)

    print("\nWeak pattern proof:")
    wp = WeakPattern()
    wp.weak_pattern_proof(X_train, X_test, Y_train, Y_test)
