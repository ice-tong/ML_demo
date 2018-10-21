# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 12:51:48 2018

@author: icetong
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 09:23:03 2018

@author: icetong
"""

from autograd import numpy as np
from autograd import grad
import pandas as pd
import pickle
import  time


class MyLogisticReg:
    """A class for Linear RegressionModels"""
    
    def __init__(self, max_iter=2000, alpha=0.001, C=0.01, epsilon=1e-4):
        self.max_iter = max_iter
        self.alpha = alpha
        self.C = C
        self.epsilon = epsilon
        self.Weights = None
    
    def _cost(self, weights, x, y):
        eta = x @ weights
        l2_term = 0.5 * self.C * np.sum(weights**2)
        cost = l2_term - np.sum(y*eta - np.log(1 + np.exp(eta)))
        return cost
    
    def _gradAscent(self, X_, Y_):
        self.gd_cost_histroy = []
        self.gd_iters = []
        self.gd_timers = []
        t = time.time()
        weights_histroy = np.zeros([100, self.Weights.shape[0],1])
        grad_cost = grad(self._cost)
        for i in range(self.max_iter):
            diff = np.mean(abs(self.Weights - weights_histroy[i%100, :]))
            if diff < self.epsilon: break
            weights_histroy[i%100, :] = self.Weights
            self.Weights = self.Weights - self.alpha * grad_cost(self.Weights, X_, Y_)
            cost = self._cost(self.Weights, X_, Y_)
            if i % 100 == 0:
                print("L(w, w0) values now: ", cost)
            self.gd_cost_histroy.append(cost)
            self.gd_iters.append(i)
            self.gd_timers.append(time.time()-t)
        return self.Weights
    
    def _stocGradAscent(self, X_, Y_, batch_size=10):
        weights_histroy = np.zeros([100, self.Weights.shape[0], 1])
        self.sgd_cost_histroy = []
        self.sgd_iters = []
        self.sgd_timers = []
        t = time.time()
        grad_cost = grad(self._cost)
        for i in range(self.max_iter):
            random_index = np.random.randint(0, X_.shape[0]-batch_size)
            x = X_[random_index:random_index+batch_size]
            y = Y_[random_index:random_index+batch_size]
            diff = np.mean(abs(self.Weights - weights_histroy[i%100, :]))
            if diff < self.epsilon: break
            weights_histroy[i%100] = self.Weights
            self.Weights = self.Weights - self.alpha * grad_cost(self.Weights, x, y)
            cost = (self.N/batch_size) * self._cost(self.Weights, x, y)
            if i % 100 == 0:
                print("L(w, w0) values now: ", cost)
            self.sgd_cost_histroy.append(cost)
            self.sgd_iters.append(i)
            self.sgd_timers.append(time.time()-t)
        return self.Weights
    
    def fit(self, X, Y, solver="SGD"):
        X_ = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)
        Y_ = np.reshape(Y, [-1, 1])
        self.N = X_.shape[0]
        self.Weights = np.ones([X_.shape[1], 1])
        if solver == "SGD":
            self._stocGradAscent(X_, Y_)
        else:
            self._gradAscent(X_, Y_)
    
    def predict(self, X):
        X_ = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)
        y_pred = np.zeros(X_.shape[0])
        h = 1.0 / (1 + np.exp(-(X_ @ self.Weights)))
        for i in range(h.shape[0]):
            if h[i] >= 0.5:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
        return y_pred

def evaluate(y_test, y_pred):
    error_rate = (np.sum(np.equal(y_test, y_pred).astype(np.float))
            / y_test.size)
    return error_rate

def load_data(name="titanic"):
    if name == "titanic":
        df = pd.read_csv("titanic_train.csv", index_col=None, dtype=float)
        label_data = df.values[:, 0]
        features_data = df.values[:, 1:]
        features_data = (features_data - 
                         features_data.mean()) / features_data.std()
    else:
        df = pd.read_csv("mnist-train.csv", index_col=None, dtype=float)
        label_data = df['label'].map(lambda x: 0 if x==8.0 else 1).values
        features_data = df.values[:, 1:] / 255.0
    return features_data, label_data

def split_data(features, labels, rate=0.8):
    train_features = features[:int(rate*features.shape[0])]
    train_labels = labels[:int(rate*labels.shape[0])]
    test_features = features[int(rate*features.shape[0]):]
    test_labels = labels[int(rate*labels.shape[0]):]
    return train_features, train_labels, test_features, test_labels

def main():
    name = "mnist"
#    name = "titanic"
    
    features_data, label_data = load_data(name)
    (train_features, train_labels, 
         test_features, test_labels) = split_data(features_data, label_data)
    
    logistic = MyLogisticReg()
    logistic.fit(train_features, train_labels, solver="SGD")
    pred = logistic.predict(test_features)
    error_rate = evaluate(pred, test_labels)
    print(error_rate)
    
    with open("{}_classifier.pkl".format(name), "wb") as f:
        pickle.dump(logistic, f)
    
    import matplotlib.pyplot as plt
    logistic.fit(train_features, train_labels, solver="")
    plt.plot(logistic.sgd_iters, logistic.sgd_cost_histroy, label="SGD")
    plt.plot(logistic.gd_iters, logistic.gd_cost_histroy, label="gradient descent")
    plt.legend()
    plt.savefig("cost-iter.png")
    plt.show()
    plt.close()
    plt.plot(logistic.sgd_timers, logistic.sgd_cost_histroy, label="SGD")
    plt.plot(logistic.gd_timers, logistic.gd_cost_histroy, label="gradient descent")
    plt.legend()
    plt.savefig("cost-timer.png")
    plt.show()
    
#    from sklearn.linear_model import LogisticRegression
#    model = LogisticRegression()
#    model.fit(train_features, train_labels)
#    pred = model.predict(test_features)
#    error_rate = evaluate(pred, test_labels)
#    print(error_rate)

if __name__=="__main__":
    main()