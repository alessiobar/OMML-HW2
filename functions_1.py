import os
import gzip
from sklearn.model_selection import KFold
from numpy import linalg
import cvxopt
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import confusion_matrix
from cvxopt import solvers
import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

import os


def get_data(path=os.getcwd(), kind='train', cl="binary"):
    cwd = path

    X_all_labels, y_all_labels = load_mnist(cwd, kind='train')

    """
    We are only interested in the items with label 1, 5 and 7.
    Only a subset of 1000 samples per class will be used.
    """
    indexLabel1 = np.where((y_all_labels == 1))
    xLabel1 = X_all_labels[indexLabel1][:1000, :].astype('float64')
    yLabel1 = y_all_labels[indexLabel1][:1000].astype('float64')

    indexLabel5 = np.where((y_all_labels == 5))
    xLabel5 = X_all_labels[indexLabel5][:1000, :].astype('float64')
    yLabel5 = y_all_labels[indexLabel5][:1000].astype('float64')

    indexLabel7 = np.where((y_all_labels == 7))
    xLabel7 = X_all_labels[indexLabel7][:1000, :].astype('float64')
    yLabel7 = y_all_labels[indexLabel7][:1000].astype('float64')

    if cl == "binary":

        yLabel5 = [-1] * len(yLabel5)
        yLabel1 = [1] * len(yLabel1)
        y = yLabel1 + yLabel5
        x = np.concatenate((xLabel1, xLabel5), axis=0)
    else:

        yLabel5 = [0] * len(yLabel5)
        yLabel1 = [1] * len(yLabel1)
        yLabel7 = [2] * len(yLabel7)
        y = yLabel1 + yLabel5 + yLabel7
        x = np.concatenate((xLabel1, xLabel5, xLabel7), axis=0)

    scaler = MinMaxScaler()
    x_norm = scaler.fit_transform(x)
    X_tr, X_tst, y_tr, y_tst = train_test_split(x_norm, y, shuffle=True, test_size=0.2, random_state=1998079)
    return X_tr, X_tst, y_tr, y_tst


def poly_kernel(x1, x2, gamma):
    return (x1.dot(x2.T) + 1) ** gamma

def rbf_kernel(x1, x2, gamma):
    return np.exp(-gamma * (linalg.norm(x1 - x2) ** 2))

class SVM:
    def __init__(self, kernel_func, C, gamma):
        self.kernel_func = kernel_func
        self.C = C
        self.gamma = gamma
        self.tol = 1e-5

    def fit(self, X, y):
        y = np.array(y)
        K = self.kernel_func(X, X, self.gamma)

        # Creating Q: Q[i, j] = y_i * y_j * K(x_i, x_j) for objective function
        Q = cvxopt.matrix(np.outer(y,y) * K)

        # Creatin (- e) for -e^Ta for objective function (already created with minus)
        e = cvxopt.matrix(np.ones(X.shape[0]) * -1)

        # Constraint of dual SVM: a^Ty = 0, b set to 0, then Aa = b
        A = cvxopt.matrix(y * 1.0, (1, X.shape[0]))
        b = cvxopt.matrix(0.0)

        # Constraint C>=alpha >= 0 in form of G*a <= h
        M1 = np.identity(X.shape[0]) * -1
        M2 = np.identity(X.shape[0])
        G = cvxopt.matrix(np.vstack((M1, M2)))
        M1 = np.zeros(X.shape[0])
        M2 = np.ones(X.shape[0]) * self.C
        h = cvxopt.matrix(np.hstack((M1, M2)))
        time_st = time.time()

        # solving QP using cvxopt.solvers.qp
        solvers.options['show_progress'] = False
        self.solution = solvers.qp(Q, e, G, h, A, b)
        self.time_opt = time.time() - time_st
        a = np.ravel(self.solution['x'])
        # We got a - lagrange multipliers. Now getting support vectors (from all non-zero lagrange multipliers)
        # Find of non-zero lagrange multipliers
        ind = np.argwhere(a > self.tol)
        self.a = a[ind]
        # subset of training sample for which lagr multipliers > 0
        self.X_sv = X[ind]
        self.Y_sv = y[ind]
        self.b = 0
        for i in range(len(self.a)):
            self.b = self.b + self.Y_sv[i] - np.sum(self.a * self.Y_sv * K[ind.transpose()[0], ind[i]])
        self.b = self.b / len(self.a)
        return self.solution, self.time_opt

    def predict(self, X):
        y_pred = np.zeros(len(X))
        for i in range(len(X)):
            pred = 0
            for a, Y_sv, X_sv in zip(self.a, self.Y_sv, self.X_sv):
                pred += a * Y_sv * self.kernel_func(X_sv, X[i], self.gamma)
            y_pred[i] = pred
        return np.sign(y_pred + self.b), y_pred

    def SMO_MVP(self, X_train, y_train):
        import random
        P = X_train.shape[0]
        y_train = np.array(y_train)
        Q = np.outer(y_train, y_train) * self.kernel_func(X_train, X_train, self.gamma)

        #Define two functions to build R_alpha, S_alpha
        r_a_func = lambda a: np.where(((y_train == 1) & (a - self.C < self.tol)) | ((a > self.tol) & (y_train == -1)))[0]
        s_a_func = lambda a: np.where(((y_train == -1) & (a - self.C < self.tol)) | ((a > self.tol) & (y_train == 1)))[0]

        #parameter initialization
        k, a_k, grad_k = 0, np.zeros(P), -np.ones(P)
        m_a, M_a = 1, -1  # for k=0
        R_a, S_a = r_a_func(a_k), s_a_func(a_k)
        time_st = time.time()
        ctl = 0

        while m_a - M_a > self.tol:
            if ctl == 100: break #control variable to limit the number of iterations
            ctl += 1

            #Build the working set W_k
            idxOfR_a = np.where((-grad_k[R_a] * y_train[R_a] - m_a < self.tol) & (-grad_k[R_a] * y_train[R_a] - m_a > -self.tol))[0]
            idxOfS_a = np.where((-grad_k[S_a] * y_train[S_a] - M_a < self.tol) & (-grad_k[S_a] * y_train[S_a] - M_a > -self.tol))[0]
            W_i = random.choice(R_a[idxOfR_a])
            temp = S_a[idxOfS_a]
            W_j = random.choice(temp[temp!=W_i])
            W_k = [W_i, W_j]

            # Find the analytical solution for a* using "Algorithm 5.A.1" from teaching notes
            a_bar, d_ij = a_k[W_k], np.ones(2)
            grad_k_bar = grad_k[W_k]
            if grad_k_bar.dot(d_ij) < self.tol and grad_k_bar.dot(d_ij) > -self.tol:
                b_star = 0
            else:
                if grad_k_bar.dot(d_ij) < -self.tol:
                    d_star = d_ij
                    b_bar = min(self.C - a_bar[0], self.C - a_bar[1])
                else:
                    d_star = -d_ij
                    b_bar = min(a_bar[0], a_bar[1])
                if b_bar < self.tol and b_bar > -self.tol:
                    b_star = 0

                elif (d_star.dot(np.sum(Q[W_k, :][:, W_k]))).dot(d_star) < self.tol and \
                        (d_star.dot(np.sum(Q[W_k, :][:, W_k]))).dot(
                            d_star) > -self.tol:
                    b_star = b_bar
                else:
                    if (d_star.dot(np.sum(Q[W_k, :][:, W_k]))).dot(d_star) > self.tol:
                        b_nv = -grad_k_bar.T.dot(d_star) / (d_star.dot(Q[W_k, :][:, W_k])).dot(d_star)
                        b_star = min(b_bar, b_nv)
            
            #Update alpha, grad_alpha, and compute m_alpha, M_alpha
            a_star = a_bar + b_star*d_star
            a_k[W_k] = a_star
            grad_k = grad_k + (a_star[0] - a_bar[0])*Q[:, W_k[0]] + (a_star[1] - a_bar[1])*Q[:, W_k[1]]
            f_a_k = 0.5*(a_star.dot(Q[W_k, :][:, W_k])).dot(a_star)-np.sum(a_star) #value of the objective function
            R_a, S_a = r_a_func(a_k), s_a_func(a_k)
            m_a, M_a = np.max(-grad_k[R_a] * y_train[R_a]), np.min(-grad_k[S_a] * y_train[S_a])
            k += 1

        a = a_k
        time_fs = time.time() - time_st
        ind = np.argwhere(a > self.tol)
        self.a = a[ind]
        self.X_sv = X_train[ind]
        self.Y_sv = y_train[ind]
        self.b = 0
        for i in range(len(self.a)):
            self.b = self.b + self.Y_sv[i] - np.sum(self.a * self.Y_sv * self.kernel_func(X_train, X_train, self.gamma)[ind.transpose()[0], ind[i]])
        self.b = self.b / len(self.a)
        return m_a - M_a, k, time_fs


def find_hyperparameters(X_tr, y_tr, kernel_func=poly_kernel, C_s=[0.0001, 0.001, 0.001, 0.1, 0.5, 1, 5],gamma_s=[1, 2, 3, 4, 5, 10]):
    kf = KFold(n_splits=5, shuffle=True, random_state=1998079)
    acc_train = np.zeros((len(C_s), len(gamma_s)))
    acc_test = np.zeros((len(C_s), len(gamma_s)))
    for i in range(len(C_s)):
        for j in range(len(gamma_s)):
            for t, (train_index, test_index) in enumerate(kf.split(X_tr)):
                svm = SVM(kernel_func, C_s[i], gamma_s[j])
                svm.fit(X_tr[train_index], np.array(y_tr)[train_index])
                y_pred_train = svm.predict(X_tr[train_index])
                y_pred_test = svm.predict(X_tr[test_index])
                acc_train[i, j] += acc(y_pred_train, np.array(y_tr)[train_index]) / kf.get_n_splits(X_tr)
                acc_test[i, j] += acc(y_pred_test, np.array(y_tr)[test_index]) / kf.get_n_splits(X_tr)
    return acc_train, acc_test, {"best gamma": gamma_s[acc_test.argmax() % len(gamma_s)],
                                 "best C": C_s[acc_test.argmax() // len(gamma_s)]}
