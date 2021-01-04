import numpy as np
from numpy.linalg import norm

class Logger:
    """
        Data logger that records all numerical information during the optimization
        Parameters:
            method: optimizer name
            trace_norm: list that records norm of gradient at each x
            iteration: number of iterations
            x_init: initial point
            minima: convergent point
    """
    def __init__(self,stochastic = False):

        self.method = None
        self.trace_norm = []
        self.x_init = None
        self.minima = None
        if stochastic:
            self.batch_size = None
            self.epoch = None
        else:
            self.iteration = None

def armijo_backtracking(obj,x,direction,gradient,**kwargs):
    """
        Armijo Line Search
        Input:
            obj: objective function
            x:  current x
            direction: search direction
            gradient: current Gradient
            Options:
                s: default = 1
                sigma: default = 0.5
                gamma: defaault = 0.1
        Output:
            alpha: stepsize that satisfies Armijo condition
    """
    # Parameter Initialize
    class Armijo_Option:
        def __init__(self,s=1, sigma=0.5, gamma=0.1):
            self.s = s
            self.sigma = sigma
            self.gamma = gamma
    opt = Armijo_Option(**kwargs)
    alpha = opt.s

    while obj(x+alpha*direction) > obj(x) + opt.gamma * alpha * np.dot(gradient,direction):
        alpha *= opt.sigma
    return alpha

class Option:
    """ Parameter container for optimizer """
    def __init__(self, tol, max_iter, print):
        """ 
            tol: stop criteria
            max_iter: max iteration
            print: whether to print
        """
        self.tol = tol
        self.max_iter = max_iter
        self.print = print

def BFGS(obj, grad, x_init, **kwargs):
    """
        BFGS Method with Armijo backtracking
        Input:
            obj: objective function
            grad: gradient function
            x_init: initial x
            Options: 
                tol: default = 1e-4
                max_iter: default = 1000
                print: default = True
                s: default = 1
                sigma: default = 0.5
                gamma: default = 0.1
        Output:
            result: Logger that records all results
    """
    # Parameter Initialize
    class BFGS_Option(Option):
        def __init__(self,**kwargs):
            super().__init__(tol = kwargs.get('tol',1e-4), max_iter = kwargs.get("max_iter",1000), print = kwargs.get("print",True))
            self.s = kwargs.get("s",1)
            self.sigma = kwargs.get("sigma",0.5)
            self.gamma = kwargs.get("gamma",0.1)
    opt = BFGS_Option(**kwargs)

    # Logger Initialize
    result = Logger()
    result.method = "BFGS Method"
    result.x_init = np.array(x_init)

    # Start Point
    if opt.print:
        print("----- BFGS Method -----")
    k=0
    x = np.array(x_init)
    gradient = grad(x)
    H = np.eye(x.size)
    while k < opt.max_iter:
        result.trace_norm.append(norm(gradient,ord=2))
        if norm(gradient,ord=2) <= opt.tol :
            break
        direction = -H @ gradient
        alpha = armijo_backtracking(obj, x, direction, gradient, s = opt.s, sigma = opt.sigma, gamma = opt.gamma)
        x_pre = x
        x = x + alpha * direction
        gradient_pre = gradient
        gradient = grad(x)
        p = x - x_pre
        q = gradient - gradient_pre
        if np.dot(p,q) <= 1e-14:
            pass
        else:
            H = H + (np.dot((p - H @ q).reshape(-1,1), p.reshape(1,-1)) + np.dot(p.reshape(-1,1), (p - H @ q).reshape(1,-1)))/np.dot(p,q) \
                  - np.dot(p - H @ q, q)/np.dot(p, q) ** 2 * np.dot(p.reshape(-1,1), p.reshape(1,-1))
        k = k+1
        if opt.print:
            print("Iter:{:d}  \t  obj:{:.4f}  \t  grad:{:.6f}".format(k, obj(x), norm(gradient,ord=2)))
    result.iteration, result.minima = k, x
    return result

def gradient_method(obj, grad, x_init, **kwargs):
    """
        Basic Gradient Method
        Input:
            obj:
            grad:
            x_init
            Options:
                tol: default = 1e-4
                max_iter: default = 5000
                print: default = True
                s: default = 1
                sigma: default = 0.5
                gamma: default = 0.1
        Output:
            results: Logger that records all results
    """
    # parameter initialize
    class GM_Option(Option):
        def __init__(self,**kwargs):
            super().__init__(tol = kwargs.get('tol',1e-4), max_iter = kwargs.get("max_iter",5000), print = kwargs.get("print",True))
            self.s = kwargs.get("s",1)
            self.sigma = kwargs.get("sigma",0.5)
            self.gamma = kwargs.get("gamma",0.1)
    opt = GM_Option(**kwargs)

    # Logger Initialize
    result = Logger()
    result.method = "Basic Gradient Method"
    result.x_init = np.array(x_init)

    # start point
    if opt.print:
        print("----- Basic Gradient Method -----")
    k=0
    x = np.array(x_init)
    while k < opt.max_iter:
        gradient = grad(x)
        result.trace_norm.append(norm(gradient, ord=2))
        if norm(gradient,ord=2) <= opt.tol:
            break
        # iterate
        direction = -gradient
        alpha = armijo_backtracking(obj, x, direction, gradient, s = opt.s, sigma = opt.sigma, gamma = opt.gamma)
        x = x + alpha * direction
        k += 1
        if opt.print:
            print("Iter:{:d}  \t  obj:{:.4f}  \t  grad:{:.6f}".format(k, obj(x), norm(gradient,ord=2)))
    result.iteration, result.minima = k, x
    return result

def AGM(obj, grad, x_init, **kwargs):
    """
        Accelerated Gradient Method
        Input:
            obj:
            grad:
            x_init:
            Options:
                tol: default = 1e-4
                max_iter: default = 5000
                print: default = True
                alpha: default = 1
                eta: default = 0.5
        Output:
            results: Logger that records all results
    """
    # parameter initialize
    class AGM_Option(Option):
        def __init__(self,**kwargs):
            super().__init__(tol = kwargs.get('tol',1e-4), max_iter = kwargs.get("max_iter",5000), print = kwargs.get("print",True))
            self.alpha = kwargs.get("alpha",1)
            self.eta = kwargs.get("eta", 0.5)
    opt = AGM_Option(**kwargs)

    # logger initialize
    result = Logger()
    result.name = "Accelerated Gradient Method"
    result.x_init = np.array(x_init)

    # start point
    if opt.print:
        print("----- Accelerated Gradient Method -----")
    k = 0
    x = np.array(x_init)
    x_prev = x
    t = 1
    t_prev = 1
    alpha = opt.alpha
    while k < opt.max_iter:
        gradient = grad(x)
        result.trace_norm.append(norm(gradient, ord=2))
        if norm(gradient, ord =2) <= opt.tol:
            break
        # iterate
        beta = (t_prev - 1) / t
        y = x + beta * (x - x_prev)
        gradient_y = grad(y)
        x_bar = y - alpha * gradient_y
        while obj(x_bar) - obj(y) > - alpha/2 * norm(gradient_y, ord=2) ** 2:      
            alpha = opt.eta * alpha
            x_bar = y - alpha * gradient_y
        t_prev = t
        t = 1 /2 * (1 + np.sqrt(1 + 4 * t ** 2))
        x_prev = x
        x = x_bar
        k += 1
        if opt.print:
            print("Iter:{:d}  \t  obj:{:.4f}  \t  grad:{:.6f}".format(k, obj(x), norm(gradient,ord=2)))
        result.iteration, result.minima = k, x
    return result

class Adam:
    """
        Adam Method
        Attribute:
            lr: default = 1e-3
            beta1: default = 0.9
            beta2: default = 0.999
            eps: default = 1e-8
            print: default = True
        Method:
            update: update model parameters
    """
    def __init__(self, **kwargs):
        # parameter initialize
        self.lr = kwargs.get("lr", 1e-3)
        self.beta1 =kwargs.get("beta1", 0.9)
        self.beta2 = kwargs.get("beta2", 0.999)
        self.eps = kwargs.get("eps", 1e-8)
        self.print = kwargs.get("print", True)
        self.t = 0
        self.s = 0
        self.r = 0

        if self.print:
            print("----- Adam Method -----")
        
    def update(self, model, minibatch_features, minibatch_labels):
        x = np.concatenate(model.parameters)
        gradient = model.grad(x, minibatch_features, minibatch_labels) / minibatch_features.shape[0]
        self.t += 1
        self.s = self.beta1 * self.s + (1 - self.beta1) * gradient
        self.r = self.beta2 * self.r + (1 - self.beta2) * gradient * gradient 
        s_corr = self.s/(1 - self.beta1 ** self.t)
        r_corr = self.r/(1 - self.beta2 ** self.t)
        x = x - self.lr * s_corr / (np.sqrt(r_corr) + self.eps)
        model.parameters = x
        if self.print:
            print("Iter:{:d}  \t  loc.grad:{:.6f}".format(self.t, norm(gradient,ord=2)))
        return

class SGD:
    """
        SGD Method
        Attribute:
            eta: default = 0.1
            decay: default = 0.001
            momemtum: default = 0.9
            print: default = True 
            lr: linear decay with lower bound = 0.01 * eta
        Method:
            update: update model parameters
    """
    def __init__(self,**kwargs):
        # parameter initialize
        self.eta = kwargs.get("eta", 0.1)
        self.decay = kwargs.get("decay", 0.001)
        self.momemtum = kwargs.get("momemtum", 0.9)
        self.print = kwargs.get("print",True)
        self.t = 0
        self.v = 0

        if self.print:
           print("----- SGD Method -----") 

    @property
    def lr(self):
        return 1 / (1 + self.decay * self.t) if 1 / (1 + self.decay * self.t) > 0.01 * self.eta else 0.01 * self.eta

    def update(self, model, minibatch_features, minibatch_labels):
        x = np.concatenate(model.parameters)
        gradient = model.grad(x, minibatch_features, minibatch_labels) / minibatch_features.shape[0]
        self.t += 1
        self.v = self.momemtum * self.v - self.lr * gradient
        x = x + self.v
        model.parameters = x
        if self.print:
            print("Iter:{:d}  \t  loc.grad:{:.6f}".format(self.t, norm(gradient,ord=2)))
        return 