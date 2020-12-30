import numpy as np
from numpy.linalg import norm
from sklearn.utils import shuffle

# Armijo Backtracking
def armijo_backtracking(obj,x,y,direction,gradient,features,labels,options):
    s = options['s']
    sigma = options['sigma']
    gamma = options['gamma']
    while(obj(x+s*direction[:-1],y+s*direction[-1],features,labels) > obj(x,y,features,labels)+gamma*s*np.dot(gradient,direction)):
        s = sigma*s
    return s

# BFGS
def BFGS(obj,grad,x_init,y_init,features,labels,options): # options include s,sigma,gamma,tol
    x = np.array(x_init)
    y = np.array(y_init)
    m,n = features.shape
    max_iter = options["max_iter"]
    tol = options["tol"]
    isprint = options['isprint']
    k=0
    norm_gradient_list = []
    H = np.eye(n+1)
    gradient = grad(x,y,features,labels)
    norm_gradient_list.append(norm(gradient,ord=2))
    while norm(gradient,ord=2) > tol and k < max_iter:
        direction = -H @ gradient
        alpha = armijo_backtracking(obj,x,y,direction,gradient,features,labels,options)
        x_pre = x
        y_pre = y
        x = x + alpha * direction[:-1]
        y = y + alpha * direction[-1]
        gradient_pre = gradient
        gradient = grad(x,y,features,labels)
        p = np.append(x,y) - np.append(x_pre,y_pre)
        q = gradient - gradient_pre
        if np.dot(p,q) <= 1e-14:
            pass
        else:
            H = H + ( np.dot((p-H@q).reshape(-1,1),p.reshape(1,-1)) + np.dot(p.reshape(-1,1),(p-H@q).reshape(1,-1)) )/np.dot(p,q) \
                  - np.dot(p-H@q,q)/np.dot(p,q)**2 * np.dot(p.reshape(-1,1),p.reshape(1,-1))
        k = k+1
        norm_gradient_list.append(norm(gradient,ord=2))
        if isprint:
            print("Iter:",k,'\t',"alpha:",alpha,'\t',"obj:",obj(x,y,features,labels),'\t','grad:',norm(gradient))
    return x,y,norm_gradient_list

# Basic Gradient Method
def gradient_method(obj, grad,x_init,y_init, features, labels, options):
    x = np.array(x_init)
    y = np.array(y_init)
    max_iter = options["max_iter"]
    tol = options["tol"]
    isprint = options['isprint']
    norm_gradient_list = []
    k=0
    gradient = grad(x,y,features,labels)
    norm_gradient_list.append(norm(gradient,ord=2))
    while norm(gradient,ord=2) > tol and k < max_iter:
        direction = -gradient
        #armijo backtracking
        alpha = armijo_backtracking(obj,x,y,direction,gradient,features,labels,options)
        x = x + alpha * direction[:-1]
        y = y + alpha * direction[-1]
        gradient = grad(x,y,features,labels)
        norm_gradient_list.append(norm(gradient,ord=2))
        k += 1
        if isprint:
            print("Iter:",k,'\t',"alpha:",alpha,'\t',"obj:",obj(x,y,features,labels),'\t','grad:',norm(gradient))
    return x,y,norm_gradient_list

# Accelerated Gradient Method
def AGM(obj,grad,x,y,feature,label,options):
    alphak = options["alpha"]
    yita = options["yita"]
    max_iter = options["max_iter"]
    tol = options["tol"]
    isprint = options['isprint']
    x_minus = np.append(x,y)
    xk = x_minus
    tk_minus = 1
    tk = 1
    grad_list = []
    gradient = grad(xk[:-1], xk[-1], feature, label)
    k = 0
    while norm(gradient) >= tol and k < max_iter:
        beta_k = (tk_minus - 1) / tk
        y = xk + beta_k * ( xk - x_minus )
        gradient_y = grad(y[:-1], y[-1], feature, label)
        x_bar_k = y - alphak * gradient_y
        while obj(x_bar_k[:-1],x_bar_k[-1],feature, label) - obj(y[:-1], y[-1], feature, label) \
                > - 1/2 * alphak*norm(gradient_y)**2 :      
            alphak = yita * alphak
            x_bar_k = y - alphak * gradient_y
        tk_minus = tk
        tk = (1 + np.sqrt(1 + 4*tk**2))/2
        x_minus = xk
        xk = x_bar_k
        gradient = grad(xk[:-1], xk[-1], feature, label)
        grad_list.append(norm(gradient))
        k += 1
        if isprint:
            print("Iter:",k,'\t',"obj:",obj(xk[:-1],xk[-1],feature,label),'\t','grad:',norm(gradient))
    return xk[:-1],xk[-1],grad_list

def FSAGM(obj,grad,x,y,feature,label,options):
    alpha = 4*feature.shape[0]/np.trace((feature.todense().T@feature.todense()))
    max_iter = options["max_iter"]
    tol = options["tol"]
    isprint = options['isprint']

    x_minus = np.append(x,y)
    xk = np.append(x,y)

    tk_minus = 1
    tk = 1

    k = 0
    gradient = grad(xk[:-1], xk[-1], feature, label)
    grad_list = []
    while norm(gradient) >= tol and k < max_iter:

        grad_list.append(gradient)
        beta_k = (tk_minus - 1) / tk
        y = xk + beta_k * (xk - x_minus)
        x_minus = xk
        xk = y - alpha * grad(xk[:-1], xk[-1], feature, label)

        tk_minus = tk
        tk = 1 / 2 * (1 + np.sqrt(1 + 4 * tk ** 2))
        gradient = grad(xk[:-1], xk[-1], feature, label)
        k += 1
        if isprint:
            print("Iter:",k,'\t',"obj:",obj(xk[:-1],xk[-1],feature,label),'\t','grad:',norm(gradient))
    return xk[:-1], xk[-1], grad_list

#LBFGS
def LBFGS(obj, grad, x_init, y_init, features, labels, options):
    x = np.array(x_init)
    y = np.array(y_init)
    m, n = features.shape
    max_iter = options["max_iter"]
    tol = options["tol"]
    isprint = options['isprint']

    iter = 0
    norm_gradient_list = []
    gradient = grad(x, y, features, labels)
    norm_gradient_list.append(norm(gradient, ord=2))

    limit_n = 5
    ss, yy = [], []

    while norm(gradient, ord=2) > tol and iter < max_iter:

        if iter == 0:
            direction = -gradient

        alpha = armijo_backtracking(
            obj, x, y, direction, gradient, features, labels, options)
        x_pre = x
        y_pre = y
        x = x + alpha * direction[:-1]
        y = y + alpha * direction[-1]

        gradient_pre = gradient
        gradient = grad(x, y, features, labels)

        if len(ss) > limit_n and len(yy) > limit_n:
            del ss[0]
            del yy[0]

        s = np.append(x, y) - np.append(x_pre, y_pre)
        y_lbfgs = gradient - gradient_pre
        ss.append(s)
        yy.append(y_lbfgs)

        qk = gradient
        k = len(ss)
        condition = np.dot(s, y_lbfgs)
        alpha_lbfgs = []

        for i in range(k):
            t = k - i - 1
            if np.dot(yy[t].T, ss[t]) < 1e-14:
                alpha_lbfgs.append(0)
                continue
            else:
                row = 1 / np.dot(yy[t].T, ss[t])
                alpha_lbfgs.append(row * np.dot(ss[t].T, qk))
                qk = qk - alpha_lbfgs[i] * yy[t]

        if condition > 1e-14:
            gamma = np.dot(s.T, y_lbfgs) / np.dot(y_lbfgs.T, y_lbfgs)
            r = gamma * qk
        else:
            r = qk

        for i in range(k):
            t = k - i - 1
            if np.dot(yy[i].T, ss[i]) < 1e-14:
                continue
            else:
                row = 1 / np.dot(yy[i].T, ss[i])
                beta = row * np.dot(yy[i].T, r)
                r = r + ss[i] * (alpha_lbfgs[t] - beta)

        direction = -r

        iter = iter + 1
        norm_gradient_list.append(norm(gradient, ord=2))
        if isprint:
            print("Iter:", iter, '\t', "alpha:", alpha, '\t', "obj:", obj(
                x, y, features, labels), '\t', 'grad:', norm(gradient))
    return x, y, norm_gradient_list

#SGD
def SGD(obj, grad, x_init, y_init, features, labels, options):
    x = np.array(x_init)
    y = np.array(y_init)
    yita = options['yita']
    tol = options['tol']
    isprint = options['isprint']
    max_iter = options["max_iter"]
    batchsize = options.get("batchsize")
    k = 0
    norm_gradient_list = []
    # stochastic gradient descent
    while k < max_iter:
        if batchsize:
            batch = np.random.choice(features.shape[0],batchsize)
            gradient = grad(x, y, features[batch], labels[batch])
        else:
            gradient = grad(x, y, features, labels)
        norm_gradient_list.append(norm(gradient, ord=2))
        if norm(gradient, ord=2) <= tol:
            break
        beta = 1 / (1 + 0.01 * k)
        alpha = yita  * beta
        x = x - alpha * gradient[:-1]
        y = y - alpha * gradient[-1]
        k += 1
        if isprint:
            print("Iter:", k,'\t', "alpha:", alpha, '\t', "obj:", obj(x, y,features,labels), '\t', 'grad:', norm(gradient))
    return x, y, norm_gradient_list

def Adam(obj, grad, x_init, y_init, features, labels, options):
    m,n = features.shape
    x = np.array(x_init)
    y = np.array(y_init)

    beta1 = options["beta1"]
    beta2 = options["beta2"]
    lr = options["lr"]
    eps =options["eps"]
    tol = options['tol']
    isprint = options['isprint']
    max_iter = options["max_iter"]
    batchsize = options.get("batchsize")
    norm_gradient_list = []
    k = 0
    s_k = np.zeros(n+1)
    v_k = np.zeros(n+1)
    while k < max_iter:
        if batchsize:
            batch = np.random.choice(features.shape[0],batchsize)
            grad_k = grad(x, y, features[batch], labels[batch])
        else:
            grad_k = grad(x, y, features, labels)
        norm_gradient_list.append(norm(grad_k, ord=2))
        if norm(grad_k, ord=2) <= tol:
            break
        v_k = beta1 * v_k + (1 - beta1) * grad_k
        s_k = beta2 * s_k + (1 - beta2) * grad_k * grad_k
        v_k_corr = v_k / (1 - beta1 ** (k + 1))
        s_k_corr = s_k / (1 - beta2 ** (k + 1))
        grad_prime = lr * v_k_corr / np.sqrt(s_k_corr + eps)

        x = x - grad_prime[:-1]
        y = y - grad_prime[-1]
        k += 1
        if isprint:
            print("Iter:", k, '\t', "obj:", obj(x, y,features,labels), '\t', 'grad:', norm(grad_k))
    return x, y, norm_gradient_list




