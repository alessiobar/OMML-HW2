# This is the run file for the second point of the project 

'''
############################
   Importing the packages
############################
'''
import os
import gzip
from sklearn.model_selection import KFold
from numpy import linalg
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import confusion_matrix
import cvxopt
from cvxopt import matrix
import time 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from time import time


'''
############################
Function to load the data
############################
'''

# This function loads the data from the MNIST
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
    
'''
############################
     Getting the data
############################
'''

def get_data (path , kind='train', cl = "binary"):

    cwd = path

    X_all_labels, y_all_labels = load_mnist(cwd, kind='train')

    """
    We are only interested in the items with label 1, 5 and 7.
    Only a subset of 1000 samples per class will be used.
    """
    indexLabel1 = np.where((y_all_labels==1))
    xLabel1 =  X_all_labels[indexLabel1][:1000,:].astype('float64')
    yLabel1 = y_all_labels[indexLabel1][:1000].astype('float64')

    indexLabel5 = np.where((y_all_labels==5))
    xLabel5 =  X_all_labels[indexLabel5][:1000,:].astype('float64')
    yLabel5 = y_all_labels[indexLabel5][:1000].astype('float64')

    indexLabel7 = np.where((y_all_labels==7))
    xLabel7 =  X_all_labels[indexLabel7][:1000,:].astype('float64')
    yLabel7 = y_all_labels[indexLabel7][:1000].astype('float64')
    
    if cl == "binary":

        yLabel5 = [-1]*len(yLabel5)
        yLabel1 = [1]*len(yLabel1)
        y = yLabel1 + yLabel5
        x = np.concatenate((xLabel1, xLabel5), axis=0)
    else:
        
        yLabel5 = [0]*len(yLabel5)
        yLabel1 = [1]*len(yLabel1)
        yLabel7 = [2]*len(yLabel7)
        y = yLabel1 + yLabel5 + yLabel7
        x = np.concatenate((xLabel1, xLabel5, xLabel7), axis=0)

    scaler = MinMaxScaler()
    x_norm = scaler.fit_transform(x)
    X_tr, X_tst, y_tr, y_tst = train_test_split(x_norm, y, shuffle = True, test_size=0.2, random_state=1998079)
    return X_tr, X_tst, np.array(y_tr), np.array(y_tst)
    
    
'''
###########################
     Polynomial kernel 
###########################
'''

# This is the function of the polynomial kernel function
def poly_kernel(x1, x2, gamma):
    return (x1.dot(x2.T)+1) ** gamma
    
    
'''
###########################
        SVM class
###########################
'''

# This is the class for our implementation of the SVM 
class SVM:

    def __init__(self, kernel_function = None, C = None, gamma = None):
        
        # The kernel function that we want to use 
        self.kernel_function = kernel_function
        
        # The parmeter gamma of the kernel function
        self.kernel_gamma = gamma
        
        # The parameter C of the model 
        self.C = C
        
        # The number of the samples in the trainig set 
        self.P = None
        
        # The epsilon of the model 
        self.eps = 1e-6    
        
        # The tolerance of the model 
        self.tol = 1e-3
        
        # The length of the working set 
        self.q = None 
        
        # The points in the working set
        self.working_set = None
        
        # the parameter of the alpha of the model 
        self.alpha = None
        
        # the intercept of the function 
        self.b = None
        
        # the parameter gradient of alpha of the model 
        self.grad_alpha = None
        
        # the -e vector for the objective function 
        self.e = None
        
        # The X_train of the input 
        self.X_train = None
        
        # The y_train of the input 
        self.y_train = None
        
        # Setting the hessian matrix 
        self.Q = None
        
        #Number of the outer iterations 
        self.n_iter = 0
        
        # Number of the iteration of the optimizer 
        self.opt_iter = 0
        
        #The start time of fitting the model 
        self.start_time = None
        
        #The end tiem of fitting the model 
        self.end_time = None
        
        # The accuracy of the model over the trainig set 
        self.trainig_accuracy = None
        
        # The accuracy of the model over the test set
        self.test_accuracy = None
        
        # The x of the support vectors
        self.X_sv = None 
        
        # The y of the support vectors 
        self.y_sv = None
        
        # Initial value of the objective function 
        self.initial_obj_value = None 
        
        # Final value of the objective function 
        self.final_obj_value = None 
        
        # Final value of the m
        self.m = None 
        
        # Final value of M
        self.M  = None 

    # Function to give the working set 
    def get_working_set(self, First_iter = True ): 
        
        # Compute the negative gradient for all of the points  
        neg_grady = -1 * (self.Q.dot(self.alpha) - 1)/self.y_train #Negative gradient divided by Y(label data)
        
        R_alpha_set = np.where((self.alpha < self.eps) & (self.y_train == +1) | (self.alpha > self.C - self.eps) & (self.y_train == -1) | (self.alpha > self.eps) & (self.alpha < self.C - self.eps))[0]
        S_alpha_set = np.where((self.alpha < self.eps) & (self.y_train == -1) | (self.alpha > self.C - self.eps) & (self.y_train == +1) | (self.alpha > self.eps) & (self.alpha < self.C - self.eps))[0]
        
        # Take q/2 from each of the sets 
        R_q_2 = sorted(zip(list(R_alpha_set), neg_grady[R_alpha_set]), reverse=True, key=lambda x: x[1])[:self.q//2]
        S_q_2 = sorted(zip(list(S_alpha_set), neg_grady[S_alpha_set]), reverse=False, key=lambda x: x[1])[:self.q//2]
        
        # The working set 
        W_k = [x[0] for x in R_q_2] + [x[0] for x in S_q_2]        
        
        # The values for the KKT condition 
        m, M = R_q_2[0][1], S_q_2[0][1] # max from the R set, min from the S set 
        
        # Returning the working set, m, M
        return W_k, m, M
        
    # Form the hessian matrix 
    def Hessian_matrix(self, X, y): 
        
        # Applying the kernel on the data
        K = self.kernel_function(X, X, self.kernel_gamma)
        
        # Form the hessian matrix 
        Q = np.outer(y, y) * K
        
        # Return the build hessian matrix 
        return Q
    
    def Opt_alpha(self): 
        
        # indices that are not in the working set 
        Not_working_set = np.delete(np.arange(self.P), self.working_set)
        
        # Setting the parameters for the cvxopt optimizer 
        P = cvxopt.matrix(self.Q[np.ix_(self.working_set,self.working_set)])
        q = cvxopt.matrix(self.alpha[Not_working_set].T.dot(self.Q[np.ix_(Not_working_set, self.working_set)]) - 1)
        G = cvxopt.matrix(np.vstack((-np.eye(self.q),np.eye(self.q))))
        h = cvxopt.matrix(np.hstack((np.zeros(self.q), np.ones(self.q) * self.C)))
        A = cvxopt.matrix(self.y_train[self.working_set].reshape(1, -1).astype(float))
        b = cvxopt.matrix(-1* self.y_train[Not_working_set].T.dot(self.alpha[Not_working_set]))
        
        sol = cvxopt.solvers.qp(P, q, G, h, A, b, options={'show_progress': False, 'abstol': 1e-13, 'reltol':1e-13, 'feastol': 1e-13})
        
        self.opt_iter += sol['iterations']
        
        return np.array(sol['x'])

    # Computes the value of the objective function
    def objective_function(self):
        return 0.5 * (self.alpha.T.dot(self.Q)).dot(self.alpha) - self.e.T.dot(self.alpha)
    
    # The implementation of the SVM with decomposition method
    def SVM_decomposition(self, X_train, y_train, q = 2): #q even
        
        # Start the timer         
        self.start_time = time()

        '''
        ############################
         Initilizing the parameters 
        ############################
        '''
        # Setting the X_train 
        self.X_train = X_train 
        
        # Setting the y_train 
        self.y_train = y_train
        
        # Setting the number of the samples 
        self.P = X_train.shape[0]
        
        # Setting the length of the working set 
        self.q = q
        
        # Build the hessian matrix for all of the data 
        self.Q = self.Hessian_matrix(self.X_train, self.y_train)
        
        # Forming the -e for the objective function
        self.e = np.ones(self.P)
        
        # Initializing the values of alpha, grad_alpha
        self.alpha, self.grad_alpha = np.zeros(self.P), -self.e 
        
        # Get the working set 
        self.working_set, m, M = self.get_working_set()
        
        self.initial_obj_value = self.objective_function()
        
        # while we didn't converge to the solution
        while m - M > self.tol:

            #print('These are the values of m, M, K:',  m ,M, self.n_iter)
            '''
            #######################
                 Optimal alpha 
            #######################
            '''
            # Getting the optimized alpha for the points in the working set
            Optimized_alpha_w = self.Opt_alpha()
            
            '''
            #####################################
                Working set's alpha update 
            #####################################
            '''
            # Here we will update the alpha for the points that are inside the working set 
            Updated_alpha = self.alpha.copy() # A copy from the previous alpha 
            self.alpha[self.working_set] = Optimized_alpha_w.flatten() # Update the alpha for the working set 
      
            '''
            ###########################
                    Working set 
            ###########################
            '''
      
            self.working_set, m, M = self.get_working_set(False)
            self.m, self.M = m, M
            self.n_iter += 1
        
        self.final_obj_value = self.objective_function()
        # Now we should find the support vectors, and the b
        self.y_train = self.y_train.reshape(-1,1) #Y array should have 2 dims as we need it to calculate the intercept
        
        support_indx = (self.alpha > self.eps) & (self.alpha < self.C).flatten() #Support vectors indices
        
        # The support vector of X
        self.X_sv = self.X_train[support_indx] #X Support Vectors
        
        # The support vector of Y
        self.y_sv = self.y_train[support_indx] #Y Support Vectors
        
        # The alpha of the support vectors 
        self.alpha = self.alpha[support_indx]
        
        # Computing the intercept of the model 
        self.b = np.mean(self.y_sv - sum(self.alpha * self.y_sv * self.kernel_function(self.X_sv,self.X_sv, self.kernel_gamma)))
        
        # To be used in the prediction function 
        self.alpha = self.alpha.reshape(-1,1)
            
        self.end_time = time()
        return 
    
    # The prediction function 
    def predict(self, X_test): 
        return np.sign(np.sum(self.kernel_function(self.X_sv, X_test, self.kernel_gamma)*self.alpha*self.y_sv,axis=0)+self.b)
    
    # The function to compute the accuracy of the model 
    def accuracy(self, y_pred, y_train):
        accuracy = (np.sum(y_train == y_pred)/y_train.shape)[0]
        self.accuracy_value = accuracy 
        return self.accuracy_value
    
    # This function will report some insights about the model 
    def report(self): 
        print('The value of C: ', self.C)
        print('The kernel used: polynomial kernel')
        print('The value of gamma: ', self.kernel_gamma)
        print('The value of q: ', self.q)
        print('The number of the function evaluations: ', self.opt_iter)
        print('The number of the outer iterations: ', self.n_iter)
        print('The difference between m - M: ', self.m - self.M)
        print('The value of the tolerance: ', self.tol)
        print('Time necessary for optimization: ', f'{self.end_time - self.start_time:.2f}', 'sec')
        print('Initial value of the objective function: ', self.initial_obj_value)
        print('Final value of the objective function: ', self.final_obj_value)

'''
############################
      Loading the data
############################
'''

path = os.getcwd()

X_train, X_test, y_train ,y_test = get_data(path)


'''
############################
       Model fitting
############################
'''
model = SVM(kernel_function = poly_kernel, C = 0.1, gamma = 1)
model.SVM_decomposition(X_train, y_train, 4)


'''
############################
	   Report
############################
'''
y_train_pred = model.predict(X_train)
print('The accuracy of the model on the trainnig set: ', f'{model.accuracy(y_train_pred, y_train):.2f}%')
y_test_pred = model.predict(X_test)
print('The accuracy of the model on the trainnig set: ', f'{model.accuracy(y_test_pred, y_test):.2f}%')
model.report()
print('The confusion matrix is: ')
print(confusion_matrix(y_test_pred, y_test))

