import torch
import torchvision
from backpack import backpack, extend
import torch.optim as optim
from backpack.extensions import MNGD
from torchsummary import summary
import time
import math

# fixing HTTPS issue on Colab
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

# torch.set_default_dtype(torch.float64)

# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 1
PLOT = False
num_classes = 10
STEP_SIZE = 0.1
alpha_lm = 10
taw = 0.01
MAX_ITER = 60000//BATCH_SIZE
torch.manual_seed(0)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Selected Device:', device)
print('BATCH_SIZE:', BATCH_SIZE)

mnist_loader = torch.utils.data.dataloader.DataLoader(
    torchvision.datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,)
            )
        ])),
    batch_size=BATCH_SIZE,
    shuffle=True
)




# ##### base model from backpack website:
# model = torch.nn.Sequential(
#     torch.nn.Conv2d(1, 20, 3, 1, padding = (1,1)),
#     # torch.nn.BatchNorm2d(2),
#     torch.nn.ReLU(),
#     torch.nn.Conv2d(20, 20, 3, 1, padding = (1,1)),
#     torch.nn.ReLU(),
#     torch.nn.Conv2d(20, 20, 3, 1, padding = (1,1)),
#     torch.nn.ReLU(),
#     torch.nn.Conv2d(20, 20, 3, 1, padding = (1,1)),
#     torch.nn.ReLU(),
#     torch.nn.Conv2d(20, 2, 3, 1, padding = (1,1)),
#     torch.nn.ReLU(),
#     torch.nn.Flatten(), 
#     torch.nn.Linear(28*28*2, 10),
#     ).to(device)



##### fully connected network. Test for linear timings.
model = torch.nn.Sequential(
    torch.nn.Flatten(), 
    torch.nn.Linear(28*28, 100),
    # torch.nn.ReLU(),
    # torch.nn.Linear(100, 100),
    # torch.nn.ReLU(),
    # torch.nn.Linear(100, 100),
    # torch.nn.ReLU(),
    # torch.nn.Linear(100, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 10)
).to(device)

summary(model, ( 1, 28, 28))

loss_function = torch.nn.CrossEntropyLoss()
loss_function_none = torch.nn.CrossEntropyLoss(reduction='none')

def get_accuracy(output, targets):
    """Helper function to print the accuracy"""
    predictions = output.argmax(dim=1, keepdim=True).view_as(targets)
    return predictions.eq(targets).float().mean().item()



extend(model)
extend(loss_function)
extend(loss_function_none)


optimizer = optim.SGD(model.parameters(), lr=STEP_SIZE)


def get_diff(A, B):
    ''' returns relative error between A and B
    '''
    # return torch.norm(A - B)/torch.norm(A)
    return torch.norm(A - B)/torch.norm(A)


def optimal_JJT(outputs, targets):
    jac_list = 0
    vjp = 0
    grads = []

   
    with backpack(MNGD()):
        loss = loss_function(outputs, targets)
        loss.backward(retain_graph=True)

    for name, param in model.named_parameters():
        mngd_vals = param.mngd
        grads.append(param.grad.reshape(1, -1))
        # print(mngd_vals)
        # print(param.grad) 
    grads = torch.cat(grads, 1)  

    return loss, grads


acc_list = []
time_list = []
loss_list = []
epoch_time_list = []
start_time= time.time()
# loss_prev = 0.
# taylor_appx_prev = 0.

for epoch in range(EPOCHS):
    start_time_epoch = time.time()
    for batch_idx, (inputs, targets) in enumerate(mnist_loader):

        # print(model._backward_hooks)

        # for child in model.children():
        #     d = child._backward_hooks
        #     for item in d:
        #         print(d[item])
        # print('CCC')

        DAMPING = alpha_lm + taw
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        accuracy = get_accuracy(outputs, targets)

        ######## calling individual function for JJT computation
        ### Our extension

        # first compute the original gradient
        optimizer.zero_grad()
        # loss = loss_function(outputs, targets)
        # loss.backward(retain_graph=True)
        # loss_org = loss.item()

        # grad_org = []
        # grad_dict = {}
        # for name, param in model.named_parameters():
        #     grad_org.append(param.grad.reshape(1, -1))
        #     grad_dict[name] = param.grad.clone()

        # grad_org = torch.cat(grad_org, 1)
        ###### now we have to compute the true fisher
        # with torch.no_grad():
        #     sampled_y = torch.multinomial(torch.nn.functional.softmax(outputs, dim=1),1).squeeze().to(device)
        
        loss_org, grad_ = optimal_JJT(outputs, targets)
        # NGD_kernel, vjp = optimal_JJT(outputs, sampled_y, grad_org, acc_test, acc_hard_test)
        # NGD_inv = torch.linalg.inv(NGD_kernel + DAMPING * torch.eye(BATCH_SIZE))
        # v = torch.matmul(NGD_inv, vjp.unsqueeze(1))

        ####### rescale v:
        # v_sc = v/(BATCH_SIZE * DAMPING)



        # last part of SMW formula
        # grad_new = []
        # for name, param in model.named_parameters():
        #     param.grad = grad_dict[name] 
        #     grad_new.append(param.grad.reshape(1, -1))
        # grad_new = torch.cat(grad_new, 1)   
        optimizer.step()
        
        
        if batch_idx % 10 == 0:
            # print('real %f appx %f first order %f' % (loss_org, taylor_appx, loss_org + STEP_SIZE *  gp))
            # print('damping:', DAMPING)
            # if batch_idx > 0:
                # print('ro:', ro)
            acc_list.append(accuracy)
            time_list.append(time.time() - start_time)
            loss_list.append(loss_org)
            
            # print('Seq vs vmap error:', get_diff(JJT_naive_seq, JJT_naive_vmap))
            # print('opt vs backpack error:', get_diff(JJT_backpack, JJT_opt))
            # print('opt vs linear error:', get_diff(JJT_opt, JJT_linear))
            # print('opt vs conv error:', get_diff(JJT_opt, JJT_conv))
            # print('opt vs blocked error:', get_diff(JJT_opt, JJT_opt_blk))
            # print('opt vs fused error:', get_diff(JJT_opt, JJT_fused))
            # print(torch.allclose(JJT_naive_seq, JJT_opt) )
            # print('Jacobian Computation Time [Sequential]:', time_seq)
            # print('Jacobian Computation Time [Optimal]:', time_opt)
            # print('Jacobian Computation Time [VMAP]:', time_vmap)
            # print('Speedup over sequential:', time_seq/ time_opt)
            print('Elapsed time:', time.time() - start_time_epoch)
            print(
                "Iteration %3.d/%d   " % (batch_idx, MAX_ITER) +
                "Minibatch Loss %.3f  " % (loss_org) +
                "Accuracy %.0f" % (accuracy * 100) + "%"
            )

        if batch_idx >= MAX_ITER:
            break
    epoch_time = time.time() - start_time_epoch
    epoch_time_list.append(epoch_time)
    print('Elapsed time for epoch %d time: %.3f' % (epoch , epoch_time))

print('Epoch times : ', epoch_time_list)
print('Time(s)      ACC.      LOSS')
for i in range(len(time_list)):
    print('%.3f, %.3f, %.3f' %(time_list[i], acc_list[i], loss_list[i].item()))


