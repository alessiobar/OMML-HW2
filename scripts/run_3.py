import numpy as np

from functions_1 import *
import numpy
from numpy import linalg
import cvxopt
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import confusion_matrix
from cvxopt import  solvers
import time
import random
random.seed(2027647)

C = 0.1
gamma = 1
X_tr, X_tst, y_tr, y_tst = get_data()
svm = SVM(poly_kernel, C, gamma)
solution = svm.SMO_MVP(X_tr, y_tr)
y_pred_train, _ = svm.predict(X_tr)
y_pred_test, _ = svm.predict(X_tst)
accuracy_train = acc(y_pred_train,y_tr)
accuracy_test = acc(y_pred_test,y_tst)
dic = {"Kernel": "polynomial", "Hyperparameter gamma": gamma, "Hyperparameter C": C,
       "Classification rate on the training set": round(accuracy_train,3), 
       "Classification rate on the test set": round(accuracy_test,3),
       "The confusion matrix": confusion_matrix(y_tst,y_pred_test),
       "Time necessary for the optimization": round(solution[2], 3),
       "Number of optimization iterations (MaxIter=100)": solution[1],
        "Difference between m(α) and M(α)": solution[0]
      }
print(dic)
