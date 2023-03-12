#关于以下配置内容在从hyp.scrach.yaml
# lr0: 0.01  # initial learning rate (SGD=1E-2, Adam=1E-3)
# lrf: 0.2  # final OneCycleLR learning rate (lr0 * lrf)
# momentum: 0.937  # SGD momentum/Adam beta1
# weight_decay: 0.0005  # optimizer weight decay 5e-4
# warmup_epochs: 3.0  # warmup epochs (fractions ok)
# warmup_momentum: 0.8  # warmup initial momentum
# warmup_bias_lr: 0.1  # warmup initial bias lr

from torch.optim import SGD
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.models import shufflenet_v2_x0_5 as testmodel
import math
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

lr0 = 0.01  # initial learning rate (SGD=1E-2, Adam=1E-3)
lrf =  0.2  # final OneCycleLR learning rate (lr0 * lrf)
momentum = 0.937  # SGD momentum/Adam beta1
weight_decay = 0.0005  # optimizer weight decay 5e-4
warmup_epochs = 3.0  # warmup epochs (fractions ok)
warmup_momentum = 0.8  # warmup initial momentum
warmup_bias_lr = 0.1  # warmup initial bias lr

epochs = 300
nb = 200 # nb是batch的个数，number of batches = 样本个数 /batch_size
nw = 1000000 #nw = max(round(hyp['warmup_epochs'] * nb), 1000) #number of warmup iterations

def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

model = testmodel()

pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
for k, v in model.named_modules():
    if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
        pg2.append(v.bias)  # biases
    if isinstance(v, nn.BatchNorm2d):
        pg0.append(v.weight)  # no decay
    elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
        pg1.append(v.weight)  # apply decay

optimizer = SGD(pg0, lr=lr0, momentum=momentum, nesterov=True)
optimizer.add_param_group({'params': pg1, 'weight_decay': weight_decay})  # add pg1 with weight_decay
optimizer.add_param_group({'params': pg2})  # add pg2 (biases)

lf = one_cycle(1, lrf, epochs)
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

lr0,lr1,lr2 ,epoch_t= [], [], [],[]
optimizer.zero_grad()
for epoch in range(0,epochs):
    for i in range(nb):
        ni = i + nb * epoch
        if ni <= nw:#warmpup 预热阶段
            xi = [0, nw]
            for j, x in enumerate(optimizer.param_groups):
                x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, 0.01 * lf(epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(ni, xi, [0.8, 0.937])




    lr = [x['lr'] for x in optimizer.param_groups]
    lr0.append(lr[0])
    lr1.append(lr[1])
    lr2.append(lr[2])

    scheduler.step()
    epoch_t.append(epoch)
# 使用plt.subplot来创建小图.
# plt.subplot(221)表示将整个图像窗口分为2行2列, 当前位置为1.
# plt.subplot(222)表示将整个图像窗口分为2行2列, 当前位置为2.
# plt.subplot(223)表示将整个图像窗口分为2行2列, 当前位置为3.
plt.figure()
plt.subplot(221)
plt.plot(epoch_t, lr0, color="r",label='learning rate 0')
plt.legend()
print(lr0)
plt.subplot(222)
plt.plot(epoch_t, lr1, color="b",label='learning rate 1')
plt.legend()

plt.subplot(223)
plt.plot(epoch_t, lr2,color="g",label='learning rate 2')
plt.legend()

plt.show()