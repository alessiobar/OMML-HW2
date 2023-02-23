from functions_1 import * 
import numpy
from numpy import linalg
import cvxopt
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import confusion_matrix
from cvxopt import  solvers
import time 

C = 0.1
gamma = 1
X_tr, X_tst, y_tr, y_tst = get_data()    
svm = SVM(poly_kernel, C, gamma)
solution, time_opt = svm.fit(X_tr, y_tr)
y_pred_train, _ = svm.predict(X_tr)
y_pred_test, _ = svm.predict(X_tst)
accuracy_train = acc(y_pred_train,y_tr)
accuracy_test = acc(y_pred_test,y_tst)
dic = {"Kernel": "polynomial", "Hyperparameter gamma": gamma, "Hyperparameter C": C,
       "Classification rate on the training set": round(accuracy_train,3), 
       "Classification rate on the test set": round(accuracy_test,3),
       "The confusion matrix": confusion_matrix(y_tst,y_pred_test),
       "Time necessary for the optimization": round(time_opt, 3),
       "Number of optimization iterations": solution['iterations'],
        "Difference between m(α) and M(α)": solution['gap']
      }
print(dic)