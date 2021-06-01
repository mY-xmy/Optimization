import numpy as np
from scipy import sparse
from numpy.linalg import norm
from numpy.linalg import inv
from sklearn.utils import shuffle
from optim import BFGS,gradient_method,AGM,Adam,SGD
import optim
import warnings
import tqdm

class AIRLS:
    """
        Matrix Factorization by Alternating IRLS Algorithm
    """
    def __init__(self, rank=5):
        self.rank = rank
    
    def init_args(self, X):
        m,n = X.shape
        self.m = m
        self.n = n
        self.U = np.random.randn(m, self.rank)
        self.V = np.random.randn(n, self.rank)
        self.w1= np.ones((m, n))
        self.w2 = np.ones((n, m))
        
    def fit(self, X, ans=None, max_iter=100):
        self.init_args(X)
        errors = []
        for i in tqdm(range(1, max_iter+1)):
            # Fix U, Optimize V
            for j in range(self.n):
                beta, w = self.IRLS(X[:, j], self.U, self.V[j], self.w2[j])
                self.V[j] = beta
                self.w2[j] = w
            # Fix V, Optimize U
            for j in range(self.m):
                beta, w = self.IRLS(X[j], self.V, self.U[j], self.w1[j])   
                self.U[j] = beta
                self.w1[j] = w
            if ans is not None:
                error = norm(ans - self.U @ self.V.T, ord="fro")
            else:
                error = norm(X - self.U @ self.V.T, ord="fro")
            errors.append(error)

        return self.U, self.V, errors

    def IRLS(self, y, X, beta, w, delta = 1e-8):
        """
            solve argmin_{beta} ||y - X dot beta ||_1
        """
        W = sparse.diags(w)
        beta = inv(X.T @ W @ X) @ X.T @ W @ y
        w = 1 / np.maximum(delta, np.abs(y - X @ beta))
        return beta, w   

class SubGM:
    """
        Matrix Factorization by SubGradient based optimization method
    """
    def __init__(self, rank=5):
        self.rank = rank
    
    def init_args(self, X):
        m,n = X.shape
        self.m = m
        self.n = n
        self.U = np.random.randn(m, self.rank)
        self.V = np.random.randn(n, self.rank)
        return self.U, self.V

    def subgrad(self,X):
        TMP = np.sign(X-self.U@(self.V).T)
        """p,q = TMP.shape
        for i in range(p):
            for j in range(q):
                if TMP[i][j] == 0:
                    TMP[i][j] = np.random.uniform(-1,1)"""
        TMP[TMP==0] = np.random.uniform(-1,1)
        sub_grad_U = TMP@(-self.V)
        sub_grad_V = (TMP).T@(-self.U)
        return sub_grad_U, sub_grad_V
    
    def fit(self, X, ans=None, alpha=1e-4, max_iter=100, diminishing=False):
        # fixed step size
        lr = alpha
        self.init_args(X)
        Loss = []
        for i in tqdm(range(0, max_iter)):       
            sub_grad_U, sub_grad_V = self.subgrad(X)
            if diminishing:
                lr = alpha / np.sqrt(i+1)
            self.U = self.U - lr * sub_grad_U
            self.V = self.V - lr * sub_grad_V
            if ans is not None:
                tmp_loss = norm(ans - self.U @ self.V.T, ord="fro")
            else:
                tmp_loss = norm(X - self.U @ self.V.T, ord="fro")
            Loss.append(tmp_loss)

        return self.U, self.V, Loss