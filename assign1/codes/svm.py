from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

from utils import *

X_train,y_train,X_test,y_test = load_data(hog=True)
y_train = np.argmax(y_train,axis=0)
y_test = np.argmax(y_test,axis=0)


def SVM():
    print('*** SVM CLassifier ***')
    clf = LinearSVC(verbose=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Classification report for - \n{}:\n{}\n".format(
        clf, metrics.classification_report(y_test, y_pred,digits=4)))
    print(metrics.confusion_matrix(y_test,y_pred))

def KNN():
    print('*** K-Nearest Neighbours ***')
    print('This may take a while...')
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train) 
    y_pred = knn.predict(X_test)
    print("Classification report for - \n{}:\n{}\n".format(
        knn, metrics.classification_report(y_test, y_pred,digits=4)))
    print(metrics.confusion_matrix(y_test,y_pred))
