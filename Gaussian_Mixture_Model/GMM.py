import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets._samples_generator import make_blobs

class GMM:
    def __init__(self, k, max_iter=100):
        self.k = k
        self.max_iter = int(max_iter)

    def initialize(self, X):
        # returns the (r,c) value of the numpy array of X
        self.shape = X.shape
        # n has the number of rows while m has the number of columns of dataset X
        self.n, self.m = self.shape

        # initial weights given to each cluster are stored in phi or P(Ci=j)
        self.phi = np.full(shape=self.k, fill_value=1 / self.k)

        # initial weights given to each data point wrt to each cluster or P(Xi/Ci=j)
        self.weights = np.full(shape=self.shape, fill_value=1 / self.k)

        # dataset is divided randomly into k parts of unequal sizes
        random_row = np.random.randint(low=0, high=self.n, size=self.k)

        # initial value of mean of k Gaussians
        self.mu = [X[row_index, :] for row_index in random_row]

        # initial value of covariance matrix of k Gaussians
        self.sigma = [np.cov(X.T) for _ in range(self.k)]
        # theta =(mu1,sigma1,mu2,simga2......muk,sigmak)

    # E-Step: update weights and phi holding mu and sigma constant
    def e_step(self, X):
        # updated weights or P(Xi/Ci=j)
        self.weights = self.predict_proba(X)
        # mean of sum of probability of all data points wrt to one cluster is new updated probability of cluster k or (phi)k
        self.phi = self.weights.mean(axis=0)

    # M-Step: update meu and sigma holding phi and weights constant
    def m_step(self, X):
        for i in range(self.k):
            weight = self.weights[:, [i]]
            total_weight = weight.sum()

            self.mu[i] = (X * weight).sum(axis=0) / total_weight
            self.sigma[i] = np.cov(X.T, aweights=(weight / total_weight).flatten(), bias=True)

    # responsible for clustering the data points correctly
    def fit(self, X):
        # initialise parameters like weights, phi, meu, sigma of all Gaussians in dataset X
        self.initialize(X)
        for iteration in range(self.max_iter):
            self.e_step(X)
            # iterate to update the value of meu and sigma as the clusters shift
            self.m_step(X)

    # predicts probability of each data point wrt each cluster
    def predict_proba(self, X):
        # Creates a n*k matrix denoting probability of each point wrt each cluster
        likelihood = np.zeros((self.n, self.k))
        for i in range(self.k):
            distribution = multivariate_normal(mean=self.mu[i], cov=self.sigma[i])
            # pdf : probability denisty function
            likelihood[:, i] = distribution.pdf(X)

        numerator = likelihood * self.phi
        denominator = numerator.sum(axis=1)[:, np.newaxis]
        weights = numerator / denominator
        return weights

    # predict function
    def predict(self, X):
        weights = self.predict_proba(X)
        # datapoint belongs to cluster with maximum probability
        # returns this value
        return np.argmax(weights, axis=1)

if __name__ == "__main__":
    data = pd.read_csv('Clustering_gmm.csv')
    # X = data.values
    # mean_X = X.mean(axis = 0)
    # std_X = X.std(axis = 0)
    # X[:, 0] = (X[:, 0] - mean_X[0]) / std_X[0]
    # X[:, 1] = (X[:, 1] - mean_X[1]) / std_X[1]
    # plt.figure(figsize=(7, 7))
    # plt.scatter(X[:, 0], X[:, 1])
    # plt.xlabel('Weight')
    # plt.ylabel('Height')
    # plt.title('Data Distribution')
    # plt.show()


    # X, y_true = make_blobs(n_samples=400, centers=4,
    #                        cluster_std=0.60, random_state=0)
    # rng = np.random.RandomState(13)
    # X = X[:, ::-1]  # flip axes for better plotting
    # X_stretched = np.dot(X, rng.randn(2, 2))
    # plt.scatter(X_stretched[:, 0], X_stretched[:, 1])
    # gmm = GMM(k=4, max_iter=100)
    # gmm.fit(X)
    #
    # labels = gmm.predict(X)
    #
    # print(labels)
    #
    # plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
    #
    # plt.show()