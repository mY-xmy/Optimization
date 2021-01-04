# Optimization
ML models and optimization methods

## 1. Model
### SVM
> class model.SVM_Huber(c=0.1, delta=0.1, max_iter=1000, tol=1e-4)

#### Parameters
- **c**: regularization parameter, default = 1e-1
- **delta**: Huber norm parameter, default = 1e-1
- **max_iter**: max iteration, default = 1000
- **tol**: tolerance, default = 1e-4

#### Attributes
- **weight**: ndarray of shape (n_features,)
- **bias**: ndarray of shape (1,)


#### Methods
- **obj**(x,features,labels): objective function of SVM
- **grad**(x,features,labels): gradient function of SVM
- **parameters**(*new_parameters): return (weight, bias) or modify the attributes
- **fit**(features,labels,opt_method, **kwargs): train model with given data and optimization method
  - kwargs: parameters for optimization methods, epoch and batch_size for stochastic methods.
- **partial_fit**(features,labels,opt_method)
- **predict**(features): return predicted labels with given features

### Logistic Regression
> class model.LogisticRegression(c=0.1, max_iter=1000, tol=1e-4)

#### Parameters
- **c**: regularization parameter, default = 1e-1
- **max_iter**: max iteration, default = 1000
- **tol**: tolerance, default = 1e-4

#### Attributes
- **weight**: ndarray of shape (n_features,)
- **bias**: ndarray of shape (1,)


#### Methods
- **obj**(x,features,labels): objective function of LR
- **grad**(x,features,labels): gradient function of LR
- **parameters**(*new_parameters): return (weight, bias) or modify the attributes
- **fit**(features,labels,opt_method): train model with given data and optimization method
- **partial_fit**(features,labels,opt_method)
- **predict**(features,threshold): return predicted labels with given features


## 2. Optim

### Logger
> class Logger(stochastic=False)

Data logger that records all numerical information during the optimization
#### Attributes
- **method**: optimizer name
- **trace_norm**: list that records norm of gradient at each x
- **iteration**: number of iterations for traditional optimization method
- **batch_size**: batch size for stochastic optimization method
- **epoch**: epoch num for stochastic optimization method
- **x_init**: initial point
- **minima**: convergent point

### Armijo Backtracking
> def armijo_backtracking(obj, x, direction, gradient, s=1, sigma=0.5, gamma=0.1):

#### Parameters
- **obj**: objective function
- **x**: current $x$
- **direction**: search direction
- **gradient**: gradient of objective function at $x$
- **s**: default = 1
- **sigma**: default = 0.5
- **gamma**: default = 0.1

### BFGS
> def BFGS(obj, grad, x_init, tol=1e-4, max_iter=1000, print=True, s=1, sigma=0.5, gamma=0.1)

#### Parameters
- **obj**: objective function
- **grad**: gradient function
- **x_init**: initial x
- **tol**: tolerance, default = 1e-4
- **max_iter**: max iteration, default = 1000
- **print**: whether to print, default = True
- **s**: default = 1
- **sigma**: default = 0.5
- **gamma**: default = 0.1

### Gradient Method
> def gradient_method(obj, grad, x_init, tol=1e-4, max_iter=1000, print=True, s=1, sigma=0.5, gamma=0.1)

#### Parameters
- **obj**: objective function
- **grad**: gradient function
- **x_init**: initial x
- **tol**: tolerance, default = 1e-4
- **max_iter**: max iteration, default = 1000
- **print**: whether to print, default = True
- **s**: default = 1
- **sigma**: default = 0.5
- **gamma**: default = 0.1

### Accelerated Gradient Method
> def AGM(obj, grad, x_init, tol=1e-4, max_iter=1000, print=True, alpha=1, eta=0.5)

#### Parameters
- **obj**: objective function
- **grad**: gradient function
- **x_init**: initial x
- **tol**: tolerance, default = 1e-4
- **max_iter**: max iteration, default = 1000
- **print**: whether to print, default = True
- **alpha**: default = 1
- **eta**: default = 0.5

### Adam
> class Adam(lr=1e-3, beta1=0.9, beta2=0.999, eps= 1e-8, print=True)

#### Parameters
- **lr**: default = 1e-3
- **beta1**: default = 0.9
- **beta2**: default = 0.999
- **eps**: default = 1e-8
- **print**: default = True


#### Methods
- **update**(model, minibatch_X, minibatch_y): using minibatch to update the model parameters

### SGD
> class SGD(eta=0.1, decay=0.001, momemtum=0.9, print=True)

#### Parameters
- eta: default = 0.1
- decay: default = 0.001
- momemtum: default = 0.9
- print: default = True

#### Attributes
- **lr**: linear decaying learning rate with lower bound = 0.01 * eta

$$lr = max(\frac{1}{1 + decay * t},0.01 * \eta)$$

#### Methods
- **update**(model, minibatch_X, minibatch_y): using minibatch to update the model parameters


## 3. Metrics
To be added...