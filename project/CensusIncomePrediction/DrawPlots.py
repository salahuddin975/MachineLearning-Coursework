import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


class DrawPlot:
    def __init__(self):
        pass

    def draw_roc_comparison_plot(self, roc_info, Y_test):
        ns_probs = [0 for _ in range(len(Y_test))]
        ns_fpr, ns_tpr, _ = roc_curve(Y_test, ns_probs)
        plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')

        for roc_value in roc_info:
            plt.plot(roc_value[1], roc_value[2], label=roc_value[0])

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.show()


    def compute_roc_curve(self, clf, test_X, test_Y):
        probs = clf.predict_proba(test_X)
        probs = probs[:, 1]

        roc_score = roc_auc_score(test_Y, probs)
        print("ROC-AUC score: ", roc_score)
        fpr, tpr, _ = roc_curve(test_Y, probs)

        return roc_score, [fpr, tpr]

