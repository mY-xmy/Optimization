import numpy as np
from numpy.linalg import norm
from sklearn.utils import shuffle
from optim import BFGS,gradient_method,AGM,Adam,SGD
import optim
import warnings

class base:
    """
        Supervised Base Model
    """
    def __init__(self,max_iter, tol):
        self.max_iter = max_iter
        self.tol = tol
    def init_args(self, features, labels):
        pass
    @property
    def parameters(self):
        pass
    @parameters.setter
    def parameters(self,new_parameters):
        pass
    def obj(self,x,features,labels):
        pass
    def grad(self,x,features,labels):
        pass

    def fit(self,features,labels,opt_method,**kwargs):
        # Parameter Initialize
        m = features.shape[1]
        self.init_args(m)
        if self.max_iter:
            kwargs["max_iter"] = self.max_iter

        def obj(x):
            return self.obj(x,features,labels)
        def grad(x):
            return self.grad(x,features,labels)
        
        # Traditional Optimization Method
        if opt_method in ["BFGS","GM","AGM"]:
            x = np.concatenate(self.parameters)
            if opt_method == 'BFGS':
                result = BFGS(obj, grad, x, **kwargs)
            elif opt_method == "GM":
                result = gradient_method(obj, grad, x, **kwargs)
            elif opt_method == "AGM":
                result = AGM(obj, grad, x, **kwargs)

            self.parameters = result.minima
            # Converge Warning
            if result.iteration >= self.max_iter and result.trace_norm[-1] > self.tol:
                warnings.warn("Maximum terations reached but the optimizer hasn't converged yet.")
        
        # Stochastic Optimization Method
        elif opt_method in ["Adam","SGD"]:
            batch_size = kwargs.get("batch_size",64)
            epoch = kwargs.get("epoch",100)
            # Logger Initialize
            result = optim.Logger(stochastic = True)
            result.x_init = np.append(self.weight, self.bias)
            result.batch_size = batch_size
            result.epoch = epoch
            #Adam
            if opt_method == "Adam":
                result.method = "Adam Method"
                optimizer = Adam(**kwargs)
            # SGD
            elif opt_method == "SGD":
                result.method = "SGD Method"
                optimizer = SGD(**kwargs)
            # batch train
            for i in range(epoch):
                shuffle_X, shuffle_y = shuffle(features,labels)
                batch_num = features.shape[0]//batch_size
                for j in range(batch_num):
                    X_batch = shuffle_X[batch_size * j: min(batch_size * (j+1), features.shape[0])]
                    y_batch = shuffle_y[batch_size * j: min(batch_size * (j+1), features.shape[0])]
                    optimizer.update(self, X_batch, y_batch)
                if optimizer.print:
                    print("Epoch:{:d}  \t  opt:{:.4f}  \t  glob.grad:{:.6f}".format(i+1, 
                        obj(np.concatenate(self.parameters)), norm(grad(np.concatenate(self.parameters)), ord = 2)))
                result.trace_norm.append(norm(grad(np.concatenate(self.parameters))))
            result.minima = np.concatenate(self.parameters)

        # ValueError
        else:
            raise ValueError("Not valid opt_method!")
        return result

class SVM(base):
    """
        SVM without kernel
        Parameters:
            c: default = 1e-1
            delta: default = 1e-1
            loss: default = "squared_hinge" optional:"hinge", "huber_hinge"
            max_iter: default = 1000
            tol: default = 1e-4
            
        Method:
            obj(x,features,labels)
            grad(x,features,labels)
            fit(features,labels,opt_method)
            predict(features)
    """
    def __init__(self,c=1e-1,delta=1e-1,loss="squared_hinge",max_iter=1000,tol = 1e-1):
        super().__init__(max_iter=max_iter, tol=tol)
        self.c = c
        self.delta = delta
        self.weight = None
        self.bias = None
        assert loss in ["hinge", "squared_hinge", "huber_hinge"]
        self.loss = loss
    
    def init_args(self,m): 
        #Initial Point
        self.weight = np.random.rand(m)
        self.bias = np.random.rand(1)
    
    @property
    def parameters(self):
        return self.weight, self.bias
    
    @parameters.setter
    def parameters(self,new_parameters):
        self.weight = new_parameters[:-1]
        self.bias = new_parameters[-1:]

    def hinge(self, x, features, labels):
        t = 1 - (labels * (features @ x[:-1] + x[-1])) # 1-y*(wx+b)
        return np.sum(np.maximum(0,t))
    
    def squared_hinge(self, x, features, labels):
        return 1/2 * self.hinge(x, features, labels) ** 2

    def huber_hinge(self, x, features, labels):
        delta = self.delta
        hinge = self.hinge(x, features, labels)
        huber = np.where(hinge > delta, hinge - delta/2, 1/(2*delta) * np.maximum(0,hinge)**2)
        return np.sum(huber)

    def penalty(self,x):
        return (self.c/2) * norm(x[:-1],ord=2)**2

    def obj(self,x,features,labels):
        if self.loss == "hinge":
            return self.hinge(x, features, labels) + self.penalty(x)
        if self.loss == "squared_hinge":
            return self.squared_hinge(x, features, labels) + self.penalty(x)
        if self.loss == "huber_hinge":
            return self.huber_hinge(x, features, labels) + self.penalty(x)
    
    def grad_hinge(self,x,features,labels):
        grad_x = np.zeros(x.size - 1)
        grad_y = np.zeros(1)
        t = 1 - (labels * (features @ x[:-1] + x[-1])) # 1-y*(wx+b)
        grad_x += - labels[t>0] @ features[t>0]
        grad_y += - labels[t>0].sum()
        return np.append(grad_x, grad_y)
    
    def grad_squared_hinge(self,x,features,labels):
        return self.hinge(x, features, labels) * self.grad_hinge(x,features,labels)
    
    def grad_huber_hinge(self, x, features, labels):
        delta = self.delta
        grad_x = 0
        grad_y = 0
        t = 1 - (labels * (features @ x[:-1] + x[-1]))
        # t>delta
        grad_x += -labels[t>delta]*features[t>delta]
        grad_y += -labels[t>delta].sum()
        # t<=delta
        grad_x += -np.maximum(0,t[t <= delta])/delta * labels[t <= delta] @ features[t <= delta]
        grad_y += -np.maximum(0,t[t <= delta])/delta @ (labels[t <= delta])
        return np.append(grad_x,grad_y)

    def grad_penalty(self,x):
        c = self.c
        return np.append(c*x[:-1],0)

    def grad(self,x,features,labels):
        if self.loss == "hinge":
            return self.grad_hinge(x, features, labels) + self.grad_penalty(x)
        if self.loss == "squared_hinge":
            return self.grad_squared_hinge(x, features, labels) + self.grad_penalty(x)
        if self.loss == "huber_hinge":
            return self.grad_huber_hinge(x, features, labels) + self.grad_penalty(x)
    
    def predict(self,features):
        ''' return predict result of features '''
        return np.sign(features @ self.weight + self.bias)


class LogisticRegression(base):
    """
        Logistic Regression
        y in {-1,1}
        Parameters:
            c: default = 1e-1
            max_iter: default = 1000
            tol: default = 1e-4
        Method:
            obj(x,features,labels)
            grad(x,features,labels)
            fit(features,labels,opt_method)
            partial_fit(features,labels,opt_method)
            predict(features,threshold=0.5)
    """
    def __init__(self, c=0.1, max_iter=1000, tol = 1e-1):
        super().__init__(max_iter=max_iter, tol=tol)
        self.c = c
        self.weight = None
        self.bias = None
    
    def init_args(self, m):
        self.weight = np.random.rand(m)
        self.bias = np.random.rand(1)
    
    @property
    def parameters(self):
        return self.weight, self.bias
    
    @parameters.setter
    def parameters(self,new_parameters):
        self.weight = new_parameters[:-1]
        self.bias = new_parameters[-1:]

    def logistic(self, x, features, labels):
        m = features.shape[0]
        return 1/m * np.sum(np.log(1 + np.exp(-labels * (features @ x[:-1] + x[-1]))))
    
    def penalty(self, x):
        return (self.c/2) * norm(x[:-1],ord=2)**2

    def obj(self, x, features, labels):
        return self.logistic(x, features, labels) + self.penalty(x)
    
    def grad_logistic(self, x, features, labels):
        m = features.shape[0]
        grad_x = 1/(m * (1 + np.exp(-labels * (features @ x[:-1] + x[-1])))) * np.exp(-labels * (features @ x[:-1] + x[-1])) * -labels @ features
        grad_y = 1/(m * (1 + np.exp(-labels * (features @ x[:-1] + x[-1])))) * np.exp(-labels * (features @ x[:-1] + x[-1])) @ -labels
        return np.append(grad_x, grad_y)

    def grad_penalty(self, x):
        return np.append(self.c * x[:-1], 0)
    
    def grad(self,x, features, labels):
        return self.grad_logistic(x, features, labels) + self.grad_penalty(x)

    @property
    def parameters(self):
        return self.weight, self.bias
    
    @parameters.setter
    def parameters(self,new_parameters):
        self.weight = new_parameters[:-1]
        self.bias = new_parameters[-1:]

    def predict(self,features, threshold = 0.5):
        ''' return predict result of features '''
        return np.where(1 / (1 + np.exp(-(features @ self.weight + self.bias))) >= threshold, 1, -1)
    
class MultiLogisticRegression(base):
    """
    Multi-class Logistic Regression
    Parameters:
        k: number of class(default=2)
        c: coefficient of regularization(default=0)
        max_iter: max iteration(default=1000)
        tol: stopping tolerance(default=1e-4)
    Method:
        obj(theta,X,y)
        grad(theta,X,y)
        fit(X,y,opt_method,**kwargs)
        predict(X)
    """
    
    def __init__(self, k=2, c=0, max_iter=1000, tol=1e-4):
        super().__init__(max_iter=max_iter, tol=tol)
        self.k = k
        self.c = c
        self.weight = None
        self.bias = None
        
    def init_args(self, m):
        # start at the origin
        k = self.k
        self.weight = np.zeros((m,k)) # shape:(m,k)
        self.bias = np.zeros((1,k)) # shape:(1,k)
        
    @property
    def parameters(self):
        return self.weight, self.bias
    @parameters.setter
    def parameters(self, new_parameters):
        self.weight = new_parameters[:-1,:]
        self.bias = new_parameters[-1:,:]
    
    def penalty(self, x):
        return (self.c/2) * norm(x, ord="fro")**2
    
    def obj(self, theta, X, y):
        z = np.exp(X @ theta[:-1] + theta[-1])
        y_hat = z / z.sum(axis=1,keepdims=True)
        return -np.mean(np.log(y_hat[range(len(y)),y])) + self.penalty(theta)
        
    def grad_penalty(self, x):
        return self.c * x

    def grad(self, theta, X, y):
        """
        n: sample num
        m: feature num
        k: class num
        """
        n, m = X.shape
        k = self.k
        
        z = np.exp(X @ theta[:-1] + theta[-1]) # shape:(n,k)
        y_hat = z / z.sum(axis=1,keepdims=True) # shape:(n,k)
        
        t_onehot = np.eye(k)[y.flatten()] # shape:(n,k)
        grad_weight = X.T @ (y_hat - t_onehot) / n # shape:(m,k)
        grad_bias = np.mean(y_hat - t_onehot, axis = 0) # shape:(k,)
        #y_hat[np.arange(n),y] -= 1
        #grad_weight = X.T @ y_hat / n
        #grad_bias = np.mean(y_hat, axis = 0)
        
        return np.vstack((grad_weight, grad_bias)) + self.grad_penalty(theta) # shape:(m+1,k)
    
    def predict(self, X):
        z = np.exp(X @ self.weight + self.bias)
        y_hat = z / z.sum(axis=1,keepdims=True)
        return np.argmax(y_hat, axis = 1)