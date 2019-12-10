from enum import Enum


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

