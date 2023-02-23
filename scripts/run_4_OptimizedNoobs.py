import os
import gzip
import numpy
from numpy import linalg
import cvxopt
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import confusion_matrix
from cvxopt import  solvers
import time 
import numpy as np
from functions_1 import * 
class one_against_all:
    def __init__(self, kernel_func, C, gamma):
        self.kernel_func = kernel_func
        self.C = C
        self.gamma = gamma
    def fit(self, X, y):
        self.svm_s = []
        self.cl = (set(y))
        iters = 0
        time_opt_s = 0
        gap = []
        for i in self.cl:
            svm = SVM(self.kernel_func, self.C, self.gamma)
            y = np.array(y)
            ind_0 = np.where((y==i))[0]
            ind_1 = np.where((y!=i))[0]
            for i in ind_0:
                y[ind_0] = 1
            for i in ind_1:
                y[ind_1] = -1
            solution, time_opt = svm.fit(X,y)
            gap.append(solution['gap'])
            iters+= solution['iterations']
            time_opt_s+=time_opt
            self.svm_s.append(svm)
        return time_opt_s, iters, np.mean(gap)
    
    def predict(self, X):
        #matrix for confidence for all classes
        preds = np.zeros((len(self.cl),X.shape[0]))
        for i in range(len(self.cl)):
            _, preds[i,:] = self.svm_s[i].predict(X)
        cl = preds.argmax(0)
        return cl    
C = 0.1
gamma = 1
X_tr, X_tst, y_tr, y_tst = get_data(cl = "multiclass")    
oaa = one_against_all(poly_kernel, C = C, gamma= gamma)
time_opt, iters, gap = oaa.fit(X_tr, y_tr)
y_pred_train = oaa.predict(X_tr)
y_pred_test = oaa.predict(X_tst)
accuracy_train = acc(y_pred_train,y_tr)
accuracy_test = acc(y_pred_test,y_tst)

dic = {"Kernel": "polynomial", "Hyperparameter gamma": gamma, "Hyperparameter C": C,
       "Classification rate on the training set": round(accuracy_train,3), 
       "Classification rate on the test set": round(accuracy_test,3),
       "The confusion matrix": confusion_matrix(y_tst,y_pred_test),
       "Time necessary for the optimization": round(time_opt, 3),
       "Number of optimization iterations": iters,
        "Difference between m(α) and M(α)": gap
      }
print(dic)