# SVGD-PyTorch

The author provides sample [code](https://github.com/dilinwang820/Stein-Variational-Gradient-Descent) using numpy and explicit gradient calculation. However, PyTorch provides autograd mechanism and we can considerably simplify author's code and make it more readable.

Main changes:
- Separate the SVGD class into methods, so that we can call iteration outside the svgd process, which is more reasonable
- Support all PyTorch optimizers, rather than only adagrad in author's code. Currently we are using Adam optimizer
- Follow traditional PyTorch training pipeline, make it easier to understand.
- Implement 1-D Gaussian Mixture example, which is not provided by the author.
- Since we are using PyTorch, it is easy to use gpu, just change device to cuda.
- Make use of torch.distributions, in which log_prob() is offered, make code more readable.

Questions remained:
- Currently, I'm not sure if the gradient in bayesian inference examples are calculated correctly. Specifically, for log(alpha) term, currently the gradient may be w.r.t alpha rather than log(alpha). However, the author's code [here](https://github.com/dilinwang820/Stein-Variational-Gradient-Descent/blob/8d8f94974e1b91384dc44991ed5ad9a26212f136/python/bayesian_nn.py#L88) or [here](https://github.com/dilinwang820/Stein-Variational-Gradient-Descent/blob/8d8f94974e1b91384dc44991ed5ad9a26212f136/python/bayesian_logistic_regression.py#L54) is really hard to understand. He does not explain what is "jacobian term", and I cannot figure out where does 1 in gradient come from. So it is hard for me to find correct gradient.
- The author uses {-1,1} for logistic regression, rathan than {1,0}, which is not a normal implementation. In the [code](https://github.com/dilinwang820/Stein-Variational-Gradient-Descent/blob/8d8f94974e1b91384dc44991ed5ad9a26212f136/python/bayesian_logistic_regression.py#L54), there even appears a term that is "shape". Maybe the author should explain why. I doubt it comes from the irregular logistic regression.
- Numpy Gamma distribution and PyTorch Gamma distribution are slightly different in that one uses (shape, scale),or (k, theta) in [wiki](https://en.wikipedia.org/wiki/Gamma_distribution), as parameter, and the other uses (shape, 1/scale), or (alpha, beta) in wiki, as parameter. The author uses numpy to implement, but seems to calculate gradient according to (alpha, beta) formula?

Any suggestion to the code, and any answer to the questions is welcomed!
