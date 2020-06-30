import math
import torch


def median(tensor):
    """
    torch.median() acts differently from np.median(). We want to simulate numpy implementation.
    """
    tensor = tensor.detach().flatten()
    tensor_max = tensor.max()[None]
    return (torch.cat((tensor, tensor_max)).median() + tensor.median()) / 2.


def kernel_rbf(inputs):
    n = inputs.shape[0]
    pairwise_distance = torch.norm(inputs[:, None] - inputs, dim=2).pow(2)
    h = median(pairwise_distance) / math.log(n)
    kernel_matrix = torch.exp(-pairwise_distance / h)
    return kernel_matrix


def get_gradient(model, inputs):
    n = inputs.size(0)
    inputs = inputs.detach().requires_grad_(True)

    log_prob = model.log_prob(inputs)
    log_prob_grad = torch.autograd.grad(log_prob.sum(), inputs)[0]

    # See https://github.com/activatedgeek/svgd/issues/1#issuecomment-649235844 for why there is a factor -0.5
    kernel = kernel_rbf(inputs)
    kernel_grad = -0.5 * torch.autograd.grad(kernel.sum(), inputs)[0]

    gradient = -(kernel.mm(log_prob_grad) + kernel_grad) / n

    return gradient
