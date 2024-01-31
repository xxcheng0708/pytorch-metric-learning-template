from transformers import get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
import torchvision
import torch
from matplotlib import pyplot as plt

model = torchvision.models.resnet18(pretrained=False)
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01, momentum=0.9,
    weight_decay=5e-4
)

epoch = 50
iters = 4
# lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=5, num_training_steps=epoch * iters)
lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=5,
                                                                    num_training_steps=epoch * iters, num_cycles=2)

lr_list = []
for _ in range(epoch):
    for _ in range(iters):
        lr_list.append(optimizer.state_dict()["param_groups"][0]["lr"])

        lr_scheduler.step()

plt.plot(list(range(len(lr_list))), lr_list)
plt.show()