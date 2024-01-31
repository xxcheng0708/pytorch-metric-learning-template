import numpy as np
import torch


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]

    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


if __name__ == '__main__':
    import pandas as pd

    for alpha in [10000.0, 10.0, 5.0, 3.0, 2.0, 1.0, 0.8, 0.5, 0.3, 0.1, 0.01]:
        data = []
        for _ in range(10000):
            data.append(np.random.beta(alpha, alpha))

        data = np.asarray(data)
        print(alpha, np.mean(data), np.std(data), data[:10])

        print(pd.Series(data).quantile(q=np.linspace(0, 1.0, 11)))