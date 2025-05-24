from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn import datasets

class SampleData:
  
  def __init__(self, n_samples, data_name):
    self.n_samples = n_samples
    self.data_name = data_name

  def _generate_half_circles(self):
    noisy_circles = datasets.make_circles(
      n_samples=self.n_samples, factor=0.5, noise=0.05
    )
    
    # noisy_circles[0] is the data X, noisy_circles[1] is the labels Y
    return noisy_circles[0], noisy_circles[1]

  def _generate_moons(self):
    noisy_moons = datasets.make_moons(
      n_samples=self.n_samples, noise=0.05,random_state=6
    )
    
    # noisy_moons[0] is the data X, noisy_moons[1] is the labels Y
    return noisy_moons[0], noisy_moons[1]

  def generate_data(self):
    if 'circle' in self.data_name.lower():
        generation = self._generate_half_circles()
    
    elif 'moon' in self.data_name.lower():
        generation = self._generate_moons()
    
    else:
        raise ValueError('Invalid data name')

    # reorganize data for cross validation
    features = generation[0]
    targets = generation[1]
    targets = targets.reshape(-1, 1)

    # Concatenate features and targets horizontally
    data = np.hstack((features, targets))

    return data

  def plot(self, X, y):

    # plot the data with different colors for different classes
    ax: plt.Axes = plt.gca()
    ax.scatter(X[:, 0], X[:, 1], c=y)
    plt.title(self.data_name)
    plt.show()
    return None

class CrossValidation:
  
  def __init__(self, data, k, model, data_name, model_name):
    self.data = data
    self.k = k
    self.model = model
    self.data_name = data_name
    self.model_name = model_name
    
  def split_data(self):
    pairs = list(zip(self.data[0], self.data[1]))
    self.k_folds = np.array_split(pairs, self.k)

  def train_test(self):
    
    # intialize 
    results = {
      'Model Name': [],
      'Train Accuracy': [],
      'Test Accuracy': [],
      'Train RMSE': [],
      'Test RMSE': [],
      'Train F1': [],
      'Test F1': [],
      'Train ROC': [],
      'Test ROC': [],
    }
    
    # for each fold
    for i in range(self.k):
      
        # use k-1 folds as training data  & evaluate model
        print(f'Training on fold {i+1}')
        train_data = np.concatenate(self.k_folds[:i] + self.k_folds[i+1:])

        # split data into features and targets
        train_features = train_data[:, :-1]
        train_targets = train_data[:, -1]
        
        self.model.fit(X=train_features, y=train_targets)
        results['Model Name'].append(self.model_name)
        train_results = self.model.evaluate(train_data)
        results['Train Accuracy'].append(train_results['accuracy'])
        results['Train RMSE'].append(train_results['rmse'])
        results['Train F1'].append(train_results['f1'])
        results['Train ROC'].append(train_results['roc'])
      
        # test & evaluate model on reamining fold
        test_data = self.k_folds[i]
        test_results = self.model.evaluate(test_data)
        results['Test Accuracy'].append(test_results['accuracy'])
        results['Test RMSE'].append(test_results['rmse'])
        results['Test F1'].append(test_results['f1'])
        results['Test ROC'].append(test_results['roc'])
    
    # output average results
    avg_results = {}
    for key in results.keys():
        if key == 'Model Name':
            avg_results[key] = results[key][0]
        else:
            avg_results[key] = np.mean(results[key])

    # plot decision boundary
    self.model.plot_decision_boundary(self.data)
      
    return avg_results
    
class SimpleMLP:
  
  def __init__(self, size_layers, act_funct='sigmoid', reg_lambda=0, bias_flag=True):
    '''
    Constructor method. Defines the characteristics of the MLP

    Arguments:
        size_layers : List with the number of Units for:
            [Input, Hidden1, Hidden2, ... HiddenN, Output] Layers.
        act_funtc   : Activation function for all the Units in the MLP
            default = 'sigmoid'
        reg_lambda: Value of the regularization parameter Lambda
            default = 0, i.e. no regularization
        bias: Indicates is the bias element is added for each layer, but the output
    '''
    self.size_layers = size_layers
    self.n_layers    = len(size_layers)
    self.act_f       = act_funct
    self.lambda_r    = reg_lambda
    self.bias_flag   = bias_flag

    # Ramdomly initialize theta (MLP weights)
    self.initialize_theta_weights()
  
  def fit(self, X, y, iterations=400, reset=False):
      '''
      Given X (feature matrix) and y (class vector)
      Updates the Theta Weights by running Backpropagation N tines
      Arguments:
          X          : Feature matrix [n_examples, n_features]
          Y          : Sparse class matrix [n_examples, classes]
          iterations : Number of times Backpropagation is performed
              default = 400
          reset      : If set, initialize Theta Weights before training
              default = False
      '''
      n_examples = y.shape[0]
    #        self.labels = np.unique(y)
    #        Y = np.zeros((n_examples, len(self.labels)))
    #        for ix_label in range(len(self.labels)):
    #            # Find examples with with a Label = lables(ix_label)
    #           ix_tmp = np.where(y == self.labels[ix_label])[0]
    #            Y[ix_tmp, ix_label] = 1

      if reset:
          self.initialize_theta_weights()
      for iteration in range(iterations):
          self.gradients = self.backpropagation(X, y)
          self.gradients_vector = self.unroll_weights(self.gradients)
          self.theta_vector = self.unroll_weights(self.theta_weights)
          self.theta_vector = self.theta_vector - self.gradients_vector
          self.theta_weights = self.roll_weights(self.theta_vector)
        
  def predict(self, X):
      '''
      Given X (feature matrix), y_hay is computed
      Arguments:
          X      : Feature matrix [n_examples, n_features]
      Output:
          y_hat  : Computed Vector Class for X
      '''
      A , Z = self.feedforward(X)
      Y_hat = A[-1]
      return Y_hat

  def initialize_theta_weights(self):
    '''
    Initialize theta_weights, initialization method depends
    on the Activation Function and the Number of Units in the current layer
    and the next layer.
    The weights for each layer as of the size [next_layer, current_layer + 1]
    '''
    self.theta_weights = []
    size_next_layers = self.size_layers.copy()
    size_next_layers.pop(0)
    for size_layer, size_next_layer in zip(self.size_layers, size_next_layers):
        if self.act_f == 'sigmoid':
            # Method presented "Understanding the difficulty of training deep feedforward neurla networks"
            # Xavier Glorot and Youshua Bengio, 2010
            epsilon = 4.0 * np.sqrt(6) / np.sqrt(size_layer + size_next_layer)
            # Weigts from a uniform distribution [-epsilon, epsion]
            if self.bias_flag:  
                theta_tmp = epsilon * ( (np.random.rand(size_next_layer, size_layer + 1) * 2.0 ) - 1)
            else:
                theta_tmp = epsilon * ( (np.random.rand(size_next_layer, size_layer) * 2.0 ) - 1)            
        elif self.act_f == 'relu':
            # Method presented in "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classfication"
            # He et Al. 2015
            epsilon = np.sqrt(2.0 / (size_layer * size_next_layer) )
            # Weigts from Normal distribution mean = 0, std = epsion
            if self.bias_flag:
                theta_tmp = epsilon * (np.random.randn(size_next_layer, size_layer + 1 ))
            else:
                theta_tmp = epsilon * (np.random.randn(size_next_layer, size_layer))                    
        self.theta_weights.append(theta_tmp)
    return self.theta_weights

  def backpropagation(self, X, Y):
    '''
    Implementation of the Backpropagation algorithm with regularization
    '''
    if self.act_f == 'sigmoid':
        g_dz = lambda x: self.sigmoid_derivative(x)
    elif self.act_f == 'relu':
        g_dz = lambda x: self.relu_derivative(x)

    n_examples = X.shape[0]
    # Feedforward
    A, Z = self.feedforward(X)

    # Backpropagation
    deltas = [None] * self.n_layers
    deltas[-1] = A[-1] - Y
    # For the second last layer to the second one
    for ix_layer in np.arange(self.n_layers - 1 - 1 , 0 , -1):
        theta_tmp = self.theta_weights[ix_layer]
        if self.bias_flag:
            # Removing weights for bias
            theta_tmp = np.delete(theta_tmp, np.s_[0], 1)
        deltas[ix_layer] = (np.matmul(theta_tmp.transpose(), deltas[ix_layer + 1].transpose() ) ).transpose() * g_dz(Z[ix_layer])

    # Compute gradients
    gradients = [None] * (self.n_layers - 1)
    for ix_layer in range(self.n_layers - 1):
        grads_tmp = np.matmul(deltas[ix_layer + 1].transpose() , A[ix_layer])
        grads_tmp = grads_tmp / n_examples
        if self.bias_flag:
            # Regularize weights, except for bias weigths
            grads_tmp[:, 1:] = grads_tmp[:, 1:] + (self.lambda_r / n_examples) * self.theta_weights[ix_layer][:,1:]
        else:
            # Regularize ALL weights
            grads_tmp = grads_tmp + (self.lambda_r / n_examples) * self.theta_weights[ix_layer]       
        gradients[ix_layer] = grads_tmp;
    return gradients

  def feedforward(self, X):
    '''
    Implementation of the Feedforward
    '''
    if self.act_f == 'sigmoid':
        g = lambda x: self.sigmoid(x)
    elif self.act_f == 'relu':
        g = lambda x: self.relu(x)

    A = [None] * self.n_layers
    Z = [None] * self.n_layers
    input_layer = X

    for ix_layer in range(self.n_layers - 1):
        n_examples = input_layer.shape[0]
        if self.bias_flag:
            # Add bias element to every example in input_layer
            input_layer = np.concatenate((np.ones([n_examples ,1]) ,input_layer), axis=1)
        A[ix_layer] = input_layer
        # Multiplying input_layer by theta_weights for this layer
        Z[ix_layer + 1] = np.matmul(input_layer,  self.theta_weights[ix_layer].transpose() )
        # Activation Function
        output_layer = g(Z[ix_layer + 1])
        # Current output_layer will be next input_layer
        input_layer = output_layer

    A[self.n_layers - 1] = output_layer
    return A, Z

  def unroll_weights(self, rolled_data):
    '''
    Unroll a list of matrices to a single vector
    Each matrix represents the Weights (or Gradients) from one layer to the next
    '''
    unrolled_array = np.array([])
    for one_layer in rolled_data:
        unrolled_array = np.concatenate((unrolled_array, one_layer.flatten("F")) )
    return unrolled_array

  def roll_weights(self, unrolled_data):
    '''
    Unrolls a single vector to a list of matrices
    Each matrix represents the Weights (or Gradients) from one layer to the next
    '''
    size_next_layers = self.size_layers.copy()
    size_next_layers.pop(0)
    rolled_list = []
    if self.bias_flag:
        extra_item = 1
    else:
        extra_item = 0
    for size_layer, size_next_layer in zip(self.size_layers, size_next_layers):
        n_weights = (size_next_layer * (size_layer + extra_item))
        data_tmp = unrolled_data[0 : n_weights]
        data_tmp = data_tmp.reshape(size_next_layer, (size_layer + extra_item), order = 'F')
        rolled_list.append(data_tmp)
        unrolled_data = np.delete(unrolled_data, np.s_[0:n_weights])
    return rolled_list

  def sigmoid(self, z):
    '''
    Sigmoid function
    z can be an numpy array or scalar
    '''
    result = 1.0 / (1.0 + np.exp(-z))
    return result

  def relu(self, z):
    '''
    Rectified Linear function
    z can be an numpy array or scalar
    '''
    if np.isscalar(z):
        result = np.max((z, 0))
    else:
        zero_aux = np.zeros(z.shape)
        meta_z = np.stack((z , zero_aux), axis = -1)
        result = np.max(meta_z, axis = -1)
    return result

  def sigmoid_derivative(self, z):
    '''
    Derivative for Sigmoid function
    z can be an numpy array or scalar
    '''
    result = self.sigmoid(z) * (1 - self.sigmoid(z))
    return result

  def relu_derivative(self, z):
    '''
    Derivative for Rectified Linear function
    z can be an numpy array or scalar
    '''
    result = 1 * (z > 0)
    return result
  
  def evaluate(self, X, y):
    # intialize
    results = {}
    y_hat = self.predict(X)

    # root mean squared error
    results['rmse'] = np.sqrt(np.mean((y - y_hat) ** 2))

    # accuracy
    results['accuracy'] = np.mean(y == y_hat)

    # f1 score
    recall = np.sum((y == 1) & (y_hat == 1)) / np.sum(y == 1)
    precision = np.sum((y == 1) & (y_hat == 1)) / np.sum(y_hat == 1)
    results['f1'] = (2 * (precision * recall)) / (precision + recall)

    # ROC
    results['roc'] = roc_auc_score(y, y_hat)

    return results

  def plot_decision_boundary(self, X, y, title):
    plot_decision_regions(X, y, clf=self)
    plt.title(title)
    plt.xlabel('input 1')
    plt.ylabel('input 2')
    plt.show()

class SVM:
  
  def __init__(self, C=1.0, kernel='rbf', sigma=0.1, alpha=1, c=0, degree=2):

    # initialize the regularization parameter
    self.C = C

    # initialize the kernel function
    if kernel == 'poly':
        self.kernel = self._polynomial_kernel
        self.c = c
        self.degree = degree
        self.alpha = alpha
    elif kernel == 'rbf':
        self.kernel = self._rbf_kernel
        self.sigma = sigma
    elif kernel == 'sigmoid':
        self.kernel = self._sigmoid_kernel
        self.alpha = alpha
        self.c = c

    else:
        self.kernel = self._linear_kernel

    # initialize the inputs and targets
    self.X = None # inputs
    self.y = None # target

    # initialize the lagrangian multipliers and bias
    self.lmbda = None # lagrangian multipliers
    self.b = 0 # bias

    # initialize the ones vector
    self.ones = None

  def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    assert X.shape[1] == Y.shape[1]
    return np.exp(-(1 / self.sigma ** 2) * np.linalg.norm(X[:, np.newaxis] - Y[np.newaxis, :], axis=2) ** 2)

  def _polynomial_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    assert X.shape[1] == Y.shape[1]
    return ((self.alpha * X).dot(Y.T) + self.c) ** self.degree

  def _sigmoid_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    assert X.shape[1] == Y.shape[1]
    return np.tanh((self.alpha * X).dot(Y.T) + self.c)

  def _linear_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return (X.dot(Y.T))

  def fit(self, X: np.ndarray, y: np.ndarray, lr=1e-3, epochs=200):

      self.X = X
      self.y = y

      # (N,)
      self.lmbda = np.ones(X.shape[0]) # lagrangian multiplier lambdas
      self.b = 0 # bias
      # (N,)
      self.ones = np.ones(X.shape[0])

      # (N,N) = (N,N) * (N,N)
      y_outer = np.outer(y, y)
      y_iy_jk_ij = y_outer * self.kernel(X, X)

      for _ in range(epochs):
          # (n,)  =    (n,)      (N,N).(N,)=(N,)
          gradient = self.ones - y_iy_jk_ij.dot(self.lmbda)

          self.lmbda = self.lmbda + lr * gradient

          #bound the lagrangian multipliers for each point between 0 and C
          self.lmbda[self.lmbda > self.C] = self.C
          self.lmbda[self.lmbda < 0] = 0

          # the lagrangian formulation of the solution
          # gain = np.sum(self.lmbda) - 0.5 * np.sum(np.outer(self.lmbda, self.lmbda) * y_iy_jk_ij

      #the points that have slack > 0 (are support vectors)
      index = np.where(self.lmbda > 0 & (self.lmbda < self.C))[0]
      # (m,)= (m,)       (n,).(n,m)= (m,)
      b_i = y[index] - (self.lmbda * y).dot(self.kernel(X, X[index]))
      # Alternative code
      # b_i = y[index] - np.sum((self.lmbda * y).reshape(-1, 1)*self.kernel(X, X[index]), axis=0)
      self.b = np.mean(b_i)

  def _decision_function(self, X) -> np.ndarray:
      return (self.lmbda * self.y).dot(self.kernel(self.X, X)) + self.b

  def predict(self, X) -> np.ndarray:
      return np.sign(self._decision_function(X))

  def evaluate(self, X, y) -> dict:

    # intialize
    results = {}
    y_hat = self.predict(X)
    
    # root mean squared error
    results['rmse'] = np.sqrt(np.mean((y - y_hat) ** 2))

    # accuracy
    results['accuracy'] = np.mean(y == y_hat)

    # f1 score
    recall = np.sum((y == 1) & (y_hat == 1)) / np.sum(y == 1)
    precision = np.sum((y == 1) & (y_hat == 1)) / np.sum(y_hat == 1)
    results['f1'] = (2 * (precision * recall)) / (precision + recall)

    # ROC
    results['roc'] = roc_auc_score(y, y_hat)

    return results

  def plot_decision_boundary(self, X, y, lmbda):

    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, alpha=.5)
  
    ax: plt.Axes = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
  
    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = self._decision_function(xy).reshape(XX.shape)
  
    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors=['b', 'g', 'r'], levels=[-1, 0, 1], alpha=0.5,
    linestyles=['--', '-', '--'], linewidths=[2.0, 2.0, 2.0])
  
    # highlight the support vectors
    ax.scatter(X[:, 0][lmbda > 0.], X[:, 1][X > 0.],
    s=50, linewidth=1, facecolors='none', edgecolors='k')
  
    plt.show()