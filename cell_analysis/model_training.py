from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np


def train_svm_model(cell_df):
    feature_df = cell_df[
        ['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
    X = np.asarray(feature_df)
    y = np.asarray(cell_df['Class'].astype('int'))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    # clf = svm.SVC(kernel='rbf')
    clf = svm.SVC(kernel='linear')  # Kernel tuyến tính
    # clf = svm.SVC(kernel='poly', degree=3)  # Kernel đa thức với bậc 3
    clf.fit(X_train, y_train)

    return clf, X_test, y_test
