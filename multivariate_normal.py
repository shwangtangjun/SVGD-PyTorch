import torch
from svgd import get_gradient
from torch.optim import Adam
from torch.distributions import MultivariateNormal
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device("cpu")
sns.set(style="darkgrid")

def main():
    model = MultivariateNormal(torch.tensor([-0.6871, 0.8010], device=device),
                               torch.tensor([[0.2260, 0.1652], [0.1652, 0.6779]], device=device))

    target_samples = model.sample([10000])
    x_init = torch.zeros([100, 2], device=device).normal_(0, 1)

    # Initial plot
    sns.kdeplot(x_init[:, 0].numpy(), x_init[:, 1].numpy(), shade=True, shade_lowest=False, cbar=True, color='r')
    sns.kdeplot(target_samples[:, 0].numpy(), target_samples[:, 1].numpy(), shade=True, shade_lowest=False, cbar=True,
                color='b')
    plt.show()

    for epoch in range(1000):
        optimizer = Adam([x_init], lr=0.1 * 0.5 ** (epoch // 100))
        optimizer.zero_grad()
        x_init.grad = get_gradient(model, x_init)
        optimizer.step()
        if epoch % 100 == 99:
            print("svgd: ", torch.mean(x_init, dim=0))

    # Final plot
    sns.kdeplot(x_init[:, 0].numpy(), x_init[:, 1].numpy(), shade=True, shade_lowest=False, cbar=True, color='r')
    sns.kdeplot(target_samples[:, 0].numpy(), target_samples[:, 1].numpy(), shade=True, shade_lowest=False, cbar=True,
                color='b')
    plt.show()


if __name__ == '__main__':
    main()
