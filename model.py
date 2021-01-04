import numpy as np
from numpy.linalg import norm
from sklearn.utils import shuffle
from optim import BFGS,gradient_method,AGM,Adam,SGD
import optim
import warnings

class model:
    def __init__(self,c,max_iter, tol):
        self.c = c
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
        self.init_args(features,labels)
        kwargs["max_iter"] = self.max_iter
        def obj(x):
            return self.obj(x,features,labels)
        def grad(x):
            return self.grad(x,features,labels)
        
        # Traditional Optimization Method
        if opt_method in ["BFGS","GM","AGM"]:
            x = np.append(self.weight, self.bias)
            if opt_method == 'BFGS':
                result = BFGS(obj, grad, x, **kwargs)
            elif opt_method == "GM":
                result = gradient_method(obj, grad, x, **kwargs)
            elif opt_method == "AGM":
                result = AGM(obj, grad, x, **kwargs)

            self.weight, self.bias = result.minima[:-1], result.minima[-1:]
            # Converge Warning
            if result.iteration >= self.max_iter and result.trace_norm[-1] > self.tol:
                warnings.warn("Maximum terations reached but the optimizer hasn't converged yet.")
        
        # Stochastic Optimization Method
        elif opt_method in ["Adam","SGD"]:
            batch_size = kwargs.get("batch_size",64)
            epoch = kwargs.get("epoch",10)
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
    
    def partial_fit(self,features,labels,opt_method,**kwargs):
        # Parameter Initialize
        if self.weight is None or self.bias is None:
            #print("Initialize args") #test
            self.init_args(features,labels)
        def obj(x):
            return self.obj(x,features,labels)
        def grad(x):
            return self.grad(x,features,labels)
        
        # Traditional Optimization Method
        if opt_method in ["BFGS","GM","AGM"]:
            x = np.append(self.weight, self.bias)
            if opt_method == 'BFGS':
                result = BFGS(obj, grad, x, **kwargs)
            elif opt_method == "GM":
                result = gradient_method(obj, grad, x, **kwargs)
            elif opt_method == "AGM":
                result = AGM(obj, grad, x, **kwargs)

            self.weight, self.bias = result.minima[:-1], result.minima[-1:]
            # Converge Warning
            if result.iteration >= self.max_iter and result.trace_norm[-1] > self.tol:
                warnings.warn("Maximum terations reached but the optimizer hasn't converged yet.")
        
        # Stochastic Optimization Method
        elif opt_method in ["Adam","SGD"]:
            batch_size = kwargs.get("batch_size",64)
            epoch = kwargs.get("epoch",10)
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

class SVM(model):
    """
        SVM without kernel
    """
    def __init__(self,c,max_iter=1000, tol = 1e-1):
        self.c = c
        self.max_iter = max_iter
        self.tol = tol
        self.weight = None
        self.bias = None
    
    def init_args(self,features,labels): 
        n = features.shape[1]
        #Initial Point
        self.weight = np.random.rand(n)
        self.bias = np.random.rand(1)
    
    @property
    def parameters(self):
        return self.weight, self.bias
    
    @parameters.setter
    def parameters(self,new_parameters):
        self.weight = new_parameters[:-1]
        self.bias = new_parameters[-1:]
        
    def obj(self,x,features,labels):
        c = self.c
        t = 1 - (labels * (features @ x[:-1] + x[-1]))
        return (c/2) * norm(x[:-1],ord=2)**2 + np.sum(np.maximum(0,t))
    
    def grad_hinge(self,x,features,labels):
        grad_x = np.zeros(x.size - 1)
        grad_y = np.zeros(1)
        t = 1 - (labels * (features @ x[:-1] + x[-1]))
        grad_x += - labels[t>0] @ features[t>0]
        grad_y += - labels[t>0].sum()
        return np.append(grad_x,grad_y)
    
    def grad_norm(self,x):
        c = self.c
        return np.append(c*x[:-1],0)

    def grad(self,x,features,labels):
        return self.grad_norm(x)+ self.grad_hinge(x,features,labels)
    
    def predict(self,features):
        ''' return predict result of features '''
        return np.sign(features @ self.weight + self.bias)

class SVM_Huber(SVM):
    """
        SVM without kernel
        using Huber norm to approximate Hinge Loss
        Parameters:
            c: default = 1e-1
            delta: default = 1e-1
            max_iter: default = 1000
            tol: default = 1e-4
        Method:
            obj(x,features,labels)
            grad(x,features,labels)
            fit(features,labels,opt_method)
            partial_fit(features,labels,opt_method)
            predict(features)
    """
    def __init__(self, c = 1e-1, delta = 1e-1, max_iter=1000, tol = 1e-4):
        super().__init__(c = c, max_iter = max_iter, tol = tol)
        self.delta = delta
    
    def init_args(self,features,labels): 
        n = features.shape[1]
        #Initial Point
        self.weight = np.random.rand(n)
        self.bias = np.random.rand(1)

    def huber(self, t):
        delta = self.delta
        return 1/(2*delta) * max(0,t)**2 if t <= delta else t- delta/2
    
    def obj(self,x,features,labels):
        c = self.c
        t = 1 - (labels * (features @ x[:-1] + x[-1]))
        return (c/2) * norm(x[:-1],ord=2)**2 + sum(map(self.huber,t))
    
    def grad_huber(self,x,features,labels):
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
    
    def grad_norm(self,x):
        c = self.c
        return np.append(c*x[:-1],0)

    def grad(self,x,features,labels):
        return self.grad_norm(x)+ self.grad_huber(x,features,labels)

class LogisticRegression(model):
    """
        Logistic Regression
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
        self.c = c
        self.max_iter = max_iter
        self.tol = tol
        self.weight = None
        self.bias = None
    
    def init_args(self, features, labels):
        n = features.shape[1]
        self.weight = np.random.rand(n)
        self.bias = np.random.rand(1)

    def obj(self, x, features, labels):
        m = features.shape[0]
        return self.c /2 * norm(x[:-1],ord=2)**2 + 1/m * np.sum(np.log(1 + np.exp(-labels * (features @ x[:-1] + x[-1]))))
    
    def grad_norm(self, x):
        return np.append(self.c * x[:-1], 0)
    
    def grad(self,x, features, labels):
        m = features.shape[0]
        grad_x = 1/(m * (1 + np.exp(-labels * (features @ x[:-1] + x[-1])))) * np.exp(-labels * (features @ x[:-1] + x[-1])) * -labels @ features
        grad_y = 1/(m * (1 + np.exp(-labels * (features @ x[:-1] + x[-1])))) * np.exp(-labels * (features @ x[:-1] + x[-1])) @ -labels
        return np.append(grad_x, grad_y) + self.grad_norm(x)

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
    
    