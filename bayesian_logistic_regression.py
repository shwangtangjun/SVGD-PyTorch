import torch
import numpy as np
import random
import scipy.io
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.distributions import Normal
from torch.distributions.gamma import Gamma
from sklearn.model_selection import train_test_split
from svgd import get_gradient

'''
Example of Bayesian Logistic Regression (the same setting as Gershman et al. 2012):
Perform Stein variational gradient descent to sample from the posterior distribution.

The observed data D = {X, y} consist of N data points. 
Each has a covariate X_n in R^d and a binary class label y_n in {0,1}.
The hidden variable theta = {w, alpha} consists of d regression coefficient w_k in R,and a precision parameter alpha in 
R+. We assume the following model:
    p(alpha) = Gamma(alpha; a, b)
    p(w_k | a) = N(w_k; 0, alpha^-1)
    p(y_n = 1| x_n, w) = 1 / (1+exp(-w^T x_n))
    
Reference: https://github.com/JamesBrofos/Stein/blob/master/examples/logistic_regression/main.py
To avoid negative values of alpha, we update log(alpha) instead.
'''

device = torch.device('cpu')


class BayesianLR:
    def __init__(self, X_train, y_train, batch_size, num_particles):
        # PyTorch Gamma is slightly different from numpy Gamma
        self.alpha_prior = Gamma(torch.tensor(1., device=device), torch.tensor(1 / 0.01, device=device))
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.num_particles = num_particles

    def log_prob(self, theta):
        model_w = theta[:, :-1]
        model_alpha = torch.exp(theta[:, -1])
        # w_prior should be decided based on current alpha (not sure)
        w_prior = Normal(0, torch.sqrt(torch.ones_like(model_alpha) / model_alpha))

        random_idx = random.sample([i for i in range(self.X_train.shape[0])], self.batch_size)
        X_batch = self.X_train[random_idx]
        y_batch = self.y_train[random_idx]

        # Reference https://github.com/wsjeon/SVGD/blob/master/derivations/bayesian_classification.pdf
        # See why we can make use of binary classification loss to represent log_p_data
        logits = torch.matmul(X_batch, model_w.t())
        y_batch_repeat = y_batch.unsqueeze(1).repeat(1, self.num_particles)  # make y the same shape as logits
        log_p_data = -BCEWithLogitsLoss(reduction='none')(logits, y_batch_repeat).sum(dim=0)

        log_p0 = w_prior.log_prob(model_w.t()).sum(dim=0) + self.alpha_prior.log_prob(model_alpha)

        log_p = log_p0 + log_p_data * (self.X_train.shape[0] / self.batch_size)  # (8) in paper
        return log_p


def test(theta, X_test, y_test):
    model_w = theta[:, :-1]
    logits = torch.matmul(X_test, model_w.t())
    prob = torch.sigmoid(logits).mean(dim=1)  # Average among outputs from different network parameters(particles)
    pred = torch.round(prob)
    acc = torch.mean((pred.eq(y_test)).float())

    print("Accuracy: {}".format(acc))


def main():
    # Prepare data
    data = scipy.io.loadmat('data/covertype.mat')
    X, y = data['covtype'][:, 1:], data['covtype'][:, 0]
    y[y == 2] = 0  # y in {1,2} -> y in {1,0}
    X = np.hstack([X, np.ones([X.shape[0], 1])])  # add constant
    n_features = X.shape[1]
    X = torch.from_numpy(X).float().to(device)
    y = torch.from_numpy(y).float().to(device)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    num_particles, batch_size = 100, 200

    # random initialization (expectation of alpha is 0.01)
    theta = torch.cat([torch.zeros([num_particles, n_features], device=device).normal_(0, 10),
                       torch.log(0.01 * torch.ones([num_particles, 1], device=device))], dim=1)

    model = BayesianLR(X_train, y_train, batch_size, num_particles)
    for epoch in range(5000):
        optimizer = Adam([theta], lr=0.1 * 0.5 ** (epoch // 1000))
        optimizer.zero_grad()
        theta.grad = get_gradient(model, theta)
        optimizer.step()
        if epoch % 100 == 0:
            test(theta, X_test, y_test)

    test(theta, X_test, y_test)


if __name__ == '__main__':
    main()
