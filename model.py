import numpy as np
from opt_method import *
from numpy.linalg import norm
from scipy import sparse
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns

class SVM:
    def __init__(self,c,delta):
        self.c = c
        self.delta = delta
    
    def init_args(self,features,labels): 
        self.m, self.n = features.shape
        #Initial Point
        self.x = np.zeros(self.n) + np.random.rand(self.n)
        self.y = np.zeros(1) + np.random.rand(1)
        
    def huber(self, t):
        delta = self.delta
        return 1/(2*delta) * max(0,t)**2 if t <= delta else t- delta/2

    def obj(self, x, y, feature,label):
        '''
        x,y are parameters of SVM hyperplane
        return the SVM objective function
        '''
        c = self.c
        
        t = 1 - (label * (feature @ x + y))
        
        return (c/2) * norm(x,ord=2)**2 + sum(map(self.huber,t))
        '''
        sum = 0
        for i in range(feature.shape[0]):
            sum += self.huber(1 - label[i]*(feature[i] @ x + y))
        return (c/2) * norm(x,ord=2)**2 + float(sum)
        '''
        
    def grad_huber(self, x, y, feature, label):
        delta = self.delta
        m,n = feature.shape
        #gradsum = np.zeros(feature.shape[1]+1)
        grad_x = 0
        grad_y = 0
        
        t = 1 - (label * (feature @ x + y))

        # t>delta
        grad_x +=  -label[t>delta]*feature[t>delta]
        grad_y += -label[t>delta].sum()
        
        # t<=delta
        grad_x += -np.maximum(0,t[t <= delta])/delta * label[t <= delta] @ feature[t <= delta]
        grad_y += -np.maximum(0,t[t <= delta])/delta @ (label[t <= delta])
        
        '''
        for i in range(feature.shape[0]):
            t = float(1 - label[i] * (feature[i] * x + y))
            if t > delta:
                #gradsum += np.append((-label[i]*feature[i]).toarray().reshape(-1), -label[i])
                grad_x += -label[i]*feature[i]
                grad_y += -label[i]
            elif t>0 and t<= delta:
                #gradsum += t*(np.append((-label[i]*feature[i]).toarray().reshape(-1), -label[i]))/delta 
                grad_x += -t/delta*(label[i]*feature[i])
                grad_y += -t/delta*(label[i])
            else:
                grad_x += 0*(label[i]*feature[i])
                grad_y += 0
        '''
        return np.append(grad_x,grad_y)
        
    def grad_norm(self, x):
        c = self.c
        return np.append(c*x,0)

    def grad(self,x, y, feature, label):
        '''
        return (n+1)-dim gradient
        '''
        delta = self.delta
        return self.grad_norm(x)+ self.grad_huber(x, y, feature, label)
    
    def fit(self,features,labels,opt_method,options,max_iter=2000):
        '''
        Train the model with features and labels using selected opt_method
        Optional opt_method: "BFGS","GM","AGM"
        '''
        self.init_args(features,labels)
        x_init,y_init = self.x,self.y
        options['max_iter'] = max_iter
        if opt_method == 'BFGS':
            x,y,norm_gradient_list = BFGS(self.obj,self.grad,x_init,y_init,features,labels,options)
        elif opt_method == "GM":
            x,y,norm_gradient_list = gradient_method(self.obj,self.grad,x_init,y_init,features,labels,options)
        elif opt_method == "AGM":
            x,y,norm_gradient_list = AGM(self.obj,self.grad,x_init,y_init,features,labels,options)
        elif opt_method == "Fixed_AGM":
            x,y,norm_gradient_list = FSAGM(self.obj,self.grad,x_init,y_init,features,labels,options)
        elif opt_method == 'SGD':
            x, y, norm_gradient_list = SGD(self.obj, self.grad, x_init, y_init, features, labels, options)
        elif opt_method == 'Adam':
            x, y, norm_gradient_list = Adam(self.obj, self.grad, x_init, y_init, features, labels, options)
        elif opt_method == "LBFGS":
            x, y, norm_gradient_list = LBFGS(self.obj, self.grad, x_init, y_init, features, labels, options)
        else:
            raise ValueError("Not valid opt_method! Please Choose BFGS,GM,AGM.")
        self.x = x
        self.y = y
        return norm_gradient_list
    def predict(self,features):
        '''
        return predict result of features
        '''
        return np.sign(features @ self.x + self.y)
    
    def acc(self,features,labels):
        '''
        return accuracy between predict results and labels
        '''
        return np.sum((labels == self.predict(features)))/labels.size
    def f1_score(self,features,labels):
        '''
        return f1 score between predict results and labels
        '''
        predict = self.predict(features)
        TP = np.sum(predict[predict == 1] == labels[labels == 1])
        FP = np.sum(predict[predict == 1] == labels[labels == -1])
        FN = np.sum(predict[predict == -1] == labels[labels == 1])
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        return 2*precision*recall/(precision + recall)

class LR:
    def __init__(self, c):
        self.c = c

    def init_args(self, features, labels):
        self.m, self.n = features.shape
        # Initial Point
        self.x = np.zeros(self.n)
        self.y = np.zeros(1)

    def h_func(self, x, y, feature, label):
        #         print(1+np.exp(-np.multiply(label,feature@x+y)))
        return 1 + np.exp(-np.multiply(label, feature @ x + y))

    def obj(self, x, y, feature, label):
        '''
        x,y are parameters of Logistic Regression
        return the Logistic objective function
        '''
        c = self.c
        m = self.m
        return (c / 2) * norm(x, ord=2) ** 2 + ((1 / m) * np.sum(np.log(self.h_func(x, y, feature, label))))

    def grad_func(self, x, y, feature, label):
        grad_x = 0
        grad_y = 0
        m = self.m
        grad_x += (1 / (m * self.h_func(x, y, feature, label)) * (
                    1 - self.h_func(x, y, feature, label)) * label @ feature)
        grad_y += (1 / (m * self.h_func(x, y, feature, label)) * (1 - self.h_func(x, y, feature, label)) @ label)
        return np.append(grad_x, grad_y)

    def grad_norm(self, x):
        c = self.c
        return np.append(c * x, 0)

    def grad(self, x, y, feature, label):
        '''
        return (n+1)-dim gradient
        '''
        return self.grad_norm(x) + self.grad_func(x, y, feature, label)

    def fit(self, features, labels, opt_method, options, max_iter=2000):
        '''
        Train the model with features and labels using selected opt_method
        Optional opt_method: "BFGS","GM","AGM"
        '''
        self.init_args(features, labels)
        x_init, y_init = self.x, self.y
        options['max_iter'] = max_iter
        if opt_method == 'BFGS':
            x, y, norm_gradient_list = BFGS(
                self.obj, self.grad, x_init, y_init, features, labels, options)
        elif opt_method == "GM":
            x, y, norm_gradient_list = gradient_method(
                self.obj, self.grad, x_init, y_init, features, labels, options)
        elif opt_method == "AGM":
            x, y, norm_gradient_list = AGM(
                self.obj, self.grad, x_init, y_init, features, labels, options)
        elif opt_method == "FSAGM":
            x, y, norm_gradient_list = FSAGM(
                self.obj, self.grad, x_init, y_init, features, labels, options)
        elif opt_method == "LBFGS":
            x, y, norm_gradient_list = LBFGS(
                self.obj, self.grad, x_init, y_init, features, labels, options)
        elif opt_method == 'SGD':
            x, y, norm_gradient_list = SGD(self.obj, self.grad, x_init, y_init, features, labels, options)
        elif opt_method == 'Adam':
            x, y, norm_gradient_list = Adam(self.obj, self.grad, x_init, y_init, features, labels, options)
        else:
            raise ValueError(
                "Not valid opt_method! Please Choose BFGS,GM,AGM.")
        self.x = x
        self.y = y
        return norm_gradient_list

    def predict(self, features):
        '''
        return predict result of features
        '''
        return np.sign(1/(1+np.exp(-(features @ self.x + self.y))) - 0.5)

    def acc(self, features, labels):
        '''
        return accuracy between predict results and labels
        '''
        return np.sum((labels == self.predict(features))) / labels.size