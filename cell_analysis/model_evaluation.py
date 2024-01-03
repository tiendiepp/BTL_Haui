import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, jaccard_score
import itertools
import matplotlib.pyplot as plt

def evaluate_model(clf, X_test, y_test):
    yhat = clf.predict(X_test)
    cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
    print(classification_report(y_test, yhat))
    print('f1_score:', f1_score(y_test, yhat, average='weighted'))
    print('jaccard_score:', jaccard_score(y_test, yhat, pos_label=2))

    return cnf_matrix

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
