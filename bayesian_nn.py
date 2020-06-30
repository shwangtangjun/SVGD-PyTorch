import math
import numpy as np
import random
import torch
from torch.optim import Adam
from torch.distributions import Normal
from torch.distributions.gamma import Gamma
from sklearn.model_selection import train_test_split
from svgd import get_gradient
import torch.nn.functional as F

'''
Sample code to reproduce results for the Bayesian neural network example.
https://arxiv.org/abs/1502.05336
p(y | W, X, gamma) = prod_i^N  Normal(y_i | f(x_i; W), gamma^-1)
p(W | lambda) = prod_i^N Normal(w_i | 0, lambda^-1)
p(gamma) = Gamma(gamma | a0, b0)
p(lambda) = Gamma(lambda | a0, b0)
    
The posterior distribution is as follows:
p(W, gamma, lambda) = p(y | W, X, gamma) p(W | lambda) p(gamma) p(lambda) 
To avoid negative values of gamma and lambda, we update log(gamma) and log(lambda) instead.

theta=[w1,b1,w2,b2,log(gamma),log(lambda)]
w1: n_features * hidden_dim
b1: hidden_dim
w2: hidden_dim * 1
b2: 1
log(gamma),log(lambda): 1, 1
Actually, every parameter has num_particles rows.

Currently, we can not directly use nn.Module and construct a network, because we have many networks, each parametrized 
by a particle.
'''

device = torch.device('cpu')


class BayesianNN:
    def __init__(self, X_train, y_train, batch_size, num_particles, hidden_dim):
        self.gamma_prior = Gamma(torch.tensor(1., device=device), torch.tensor(1 / 0.1, device=device))
        self.lambda_prior = Gamma(torch.tensor(1., device=device), torch.tensor(1 / 0.1, device=device))
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.num_particles = num_particles
        self.n_features = X_train.shape[1]
        self.hidden_dim = hidden_dim

    def forward(self, inputs, theta):
        # Unpack theta
        w1 = theta[:, 0:self.n_features * self.hidden_dim].reshape(-1, self.n_features, self.hidden_dim)
        b1 = theta[:, self.n_features * self.hidden_dim:(self.n_features + 1) * self.hidden_dim].unsqueeze(1)
        w2 = theta[:, (self.n_features + 1) * self.hidden_dim:(self.n_features + 2) * self.hidden_dim].unsqueeze(2)
        b2 = theta[:, -3].reshape(-1, 1, 1)
        # log_gamma, log_lambda = theta[-2], theta[-1]

        # num_particles times of forward
        inputs = inputs.unsqueeze(0).repeat(self.num_particles, 1, 1)
        inter = F.relu(torch.bmm(inputs, w1) + b1)
        out = torch.bmm(inter, w2) + b2
        out = out.squeeze()
        return out

    def log_prob(self, theta):
        model_gamma = torch.exp(theta[:, -2])
        model_lambda = torch.exp(theta[:, -1])
        model_w = theta[:, :-2]
        # w_prior should be decided based on current lambda (not sure)
        w_prior = Normal(0, torch.sqrt(torch.ones_like(model_lambda) / model_lambda))

        random_idx = random.sample([i for i in range(self.X_train.shape[0])], self.batch_size)
        X_batch = self.X_train[random_idx]
        y_batch = self.y_train[random_idx]

        outputs = self.forward(X_batch, theta)  # [num_particles, batch_size]
        model_gamma_repeat = model_gamma.unsqueeze(1).repeat(1, self.batch_size)
        y_batch_repeat = y_batch.unsqueeze(0).repeat(self.num_particles, 1)
        distribution = Normal(outputs, torch.sqrt(torch.ones_like(model_gamma_repeat) / model_gamma_repeat))
        log_p_data = distribution.log_prob(y_batch_repeat).sum(dim=1)

        log_p0 = w_prior.log_prob(model_w.t()).sum(dim=0) + self.gamma_prior.log_prob(
            model_gamma) + self.lambda_prior.log_prob(model_lambda)
        log_p = log_p0 + log_p_data * (self.X_train.shape[0] / self.batch_size)  # (8) in paper
        return log_p


def test(model, theta, X_test, y_test):
    prob = model.forward(X_test, theta)
    y_pred = prob.mean(dim=0)  # Average among outputs from different network parameters(particles)

    print(y_pred)
    print(y_test)
    rmse = torch.norm(y_pred - y_test) / math.sqrt(y_test.shape[0])

    print("RMSE: {}".format(rmse))


def main():
    data = np.loadtxt('data/boston_housing')

    X, y = data[:, :-1], data[:, -1]
    X = torch.from_numpy(X).float().to(device)
    y = torch.from_numpy(y).float().to(device)

    # Normalization
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train_mean, X_train_std = torch.mean(X_train, dim=0), torch.std(X_train, dim=0)
    y_train_mean, y_train_std = torch.mean(y_train, dim=0), torch.std(y_train, dim=0)
    X_train = (X_train - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std
    y_train = (y_train - y_train_mean) / y_train_std
    y_test = (y_test - y_train_mean) / y_train_std

    num_particles, batch_size, hidden_dim = 100, 200, 50

    model = BayesianNN(X_train, y_train, batch_size, num_particles, hidden_dim)

    # Random initialization (based on expectation of gamma distribution)
    theta = torch.cat(
        [torch.zeros([num_particles, (X.shape[1] + 2) * hidden_dim + 1], device=device).normal_(0, math.sqrt(10)),
         torch.log(0.1 * torch.ones([num_particles, 2], device=device))], dim=1)

    for epoch in range(2000):
        optimizer = Adam([theta], lr=0.1)
        optimizer.zero_grad()
        theta.grad = get_gradient(model, theta)
        optimizer.step()
        if epoch % 100 == 0:
            test(model, theta, X_test, y_test)

    test(model, theta, X_test, y_test)


if __name__ == '__main__':
    main()
