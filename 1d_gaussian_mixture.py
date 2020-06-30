import torch
from svgd import get_gradient
from torch.optim import Adam
from torch.distributions import Normal, Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device("cpu")
sns.set(style="darkgrid")


def main():
    mix = Categorical(torch.tensor([1 / 3, 2 / 3]))
    comp = Normal(torch.tensor([-2., 2.]), torch.tensor([1., 1.]))
    model = MixtureSameFamily(mix, comp)

    target_samples = model.sample([10000])
    x_init = torch.randn([100, 1], device=device).normal_(-10, 1)

    # Initial plot
    sns.kdeplot(x_init.squeeze().numpy(), bw=0.3)
    sns.kdeplot(target_samples.squeeze().numpy(), bw=0.3)
    plt.show()

    for epoch in range(1000):
        optimizer = Adam([x_init], lr=0.1 * 0.5 ** (epoch // 250))
        optimizer.zero_grad()
        x_init.grad = get_gradient(model, x_init)
        optimizer.step()
        if epoch % 100 == 99:
            print("svgd: ", torch.mean(x_init, dim=0))

    # Final plot
    sns.kdeplot(x_init.squeeze().numpy(), bw=0.3)
    sns.kdeplot(target_samples.squeeze().numpy(), bw=0.3)
    plt.show()


if __name__ == '__main__':
    main()
