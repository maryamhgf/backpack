import torch
import torchvision
from backpack import backpack, extend
import torch.optim as optim
from backpack.extensions import Fisher, BatchGrad
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
BATCH_SIZE = 64
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




##### base model from backpack website:
model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 2, 3, 1, padding = (1,1)),
    # torch.nn.BatchNorm2d(2),
    torch.nn.ReLU(),
    torch.nn.Flatten(), 
    torch.nn.Linear(28*28*2, 10),
    ).to(device)



##### fully connected network. Test for linear timings.
# model = torch.nn.Sequential(
#     torch.nn.Flatten(), 
#     torch.nn.Linear(28*28, 100),
#     torch.nn.ReLU(),
#     torch.nn.Linear(100, 100),
#     torch.nn.ReLU(),
#     torch.nn.Linear(100, 100),
#     torch.nn.ReLU(),
#     torch.nn.Linear(100, 100),
#     torch.nn.ReLU(),
#     torch.nn.Linear(100, 10)
# ).to(device)

summary(model, ( 1, 28, 28))

loss_function = torch.nn.CrossEntropyLoss()
loss_function_none = torch.nn.CrossEntropyLoss(reduction='none')

def get_accuracy(output, targets):
    """Helper function to print the accuracy"""
    predictions = output.argmax(dim=1, keepdim=True).view_as(targets)
    return predictions.eq(targets).float().mean().item()


# class FisherOptimizer(torch.optim.Optimizer):
#     def __init__(self, parameters, step_size, damping):
#         super().__init__(
#             parameters, 
#             dict(step_size=step_size, damping=damping)
#         )

#     def step(self):
#         for group in self.param_groups:
#             print(len(group))
#             for p in group["params"]:
#                 print('p shape:',  p.shape)
#                 print('p grad shape:',  p.grad.shape)
#                 print('p fisher:',  p.fisher.shape)
#                 step_direction = p.grad / (p.fisher+ group["damping"])
#                 p.data.add_(-group["step_size"], step_direction)
#         return loss

extend(model)
extend(loss_function)
extend(loss_function_none)

# optimizer = FisherOptimizer(
#     model.parameters(), 
#     step_size=STEP_SIZE, 
#     damping=DAMPING
# )

optimizer = optim.SGD(model.parameters(), lr=STEP_SIZE)


def get_diff(A, B):
    ''' returns relative error between A and B
    '''
    # return torch.norm(A - B)/torch.norm(A)
    return torch.norm(A - B)/torch.norm(A)


def optimal_JJT(outputs, targets, grad_org, acc_test=False, acc_hard_test=False):
    jac_list = 0
    batch_grad_kernel = 0
    batch_grad_list = []
    vjp = 0
    loop_grad_kernel = 0
    loop_grad_list = []
    # note acc_test is useful when we don't have batchnorm
    # in case of batchnorm backpack fails and we need a for loop for individual grads
    
    if acc_test:
        with backpack(Fisher(), BatchGrad()):
            loss = loss_function(outputs, targets)
            loss.backward(retain_graph=True)
    else:
        with backpack(Fisher()):
            loss = loss_function(outputs, targets)
            loss.backward(retain_graph=True)

    for name, param in model.named_parameters():
        fisher_vals = param.fisher
        jac_list += fisher_vals[0]
        vjp += fisher_vals[1]
        if acc_test:
            batch_grad = BATCH_SIZE * param.grad_batch.reshape(BATCH_SIZE, -1)
            batch_grad_list.append(batch_grad)
            batch_grad_kernel += torch.matmul(batch_grad, batch_grad.t())
            param.grad_batch = None
        param.fisher = None
        # param.grad = None

    for name, param in model.named_parameters():
        param.grad = None

    if acc_hard_test:
        for i in range(BATCH_SIZE):
            loop_grad_inner_list = []

            loss = loss_function(outputs[i, :].unsqueeze(0), targets[i].unsqueeze(0))
            loss.backward(retain_graph=True)
            for name, param in model.named_parameters():
                loop_grad =  param.grad.reshape(1, -1)
                loop_grad_inner_list.append(loop_grad)
                param.grad = None
            # print(loop_grad_list)
            loop_grad = torch.cat(loop_grad_inner_list, 1)
            loop_grad_list.append(loop_grad)
        loop_grad_all = torch.cat(loop_grad_list, 0)
        loop_grad_kernel += torch.matmul(loop_grad_all, loop_grad_all.t())
    

    JJT_backpack = batch_grad_kernel / BATCH_SIZE
    JJT = jac_list / BATCH_SIZE
    JJT_loop = loop_grad_kernel / BATCH_SIZE
    if acc_test:
        all_grad = torch.cat(batch_grad_list, 1)
        backpack_vjp = torch.matmul(all_grad, grad_org.t()).view_as(vjp)
        print('NGD kernel estimation error:', get_diff(JJT_backpack, JJT))

        if get_diff(JJT_backpack, JJT) > 0.2:
            print('JJT_backpack:\n', JJT_backpack)
            print('JJT:\n', JJT)
        print('Vector Jacobian error:', get_diff(backpack_vjp, vjp))

    if acc_hard_test:
        print('NGD kernel estimation error with loop:', get_diff(JJT_loop, JJT))

    return JJT, vjp


acc_list = []
time_list = []
loss_list = []
epoch_time_list = []
start_time= time.time()
loss_prev = 0.
taylor_appx_prev = 0.
for epoch in range(EPOCHS):
    start_time_epoch = time.time()
    for batch_idx, (inputs, targets) in enumerate(mnist_loader):

        DAMPING = alpha_lm + taw
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        accuracy = get_accuracy(outputs, targets)

        ######## calling individual function for JJT computation
        ### Our extension

        # first compute the original gradient
        acc_test = True
        acc_hard_test = True
        optimizer.zero_grad()
        loss = loss_function(outputs, targets)
        loss.backward(retain_graph=True)
        loss_org = loss.item()

        grad_org = []
        grad_dict = {}
        for name, param in model.named_parameters():
            grad_org.append(param.grad.reshape(1, -1))
            grad_dict[name] = param.grad.clone()

        grad_org = torch.cat(grad_org, 1)
        ###### now we have to compute the true fisher
        with torch.no_grad():
            sampled_y = torch.multinomial(torch.nn.functional.softmax(outputs, dim=1),1).squeeze().to(device)
            
        NGD_kernel, vjp = optimal_JJT(outputs, sampled_y, grad_org, acc_test, acc_hard_test)
        NGD_inv = torch.linalg.inv(NGD_kernel + DAMPING * torch.eye(BATCH_SIZE))
        v = torch.matmul(NGD_inv, vjp.unsqueeze(1))

        ####### rescale v:
        v_sc = v/(BATCH_SIZE * DAMPING)

        # plotting NGD kernel for some iterations
        if PLOT and batch_idx in [2, 10, 50, 500] :

            JJT_opt, JJT_linear, JJT_conv = optimal_JJT() 
            
            fig, ax = plt.subplots()
            im = ax.imshow(JJT_opt , cmap='viridis')
            fig.colorbar(im,  orientation='horizontal')
            plt.show()

            fig, ax = plt.subplots()
            im = ax.imshow(JJT_linear , cmap='viridis')
            fig.colorbar(im,  orientation='horizontal')
            plt.show()

            fig, ax = plt.subplots()
            im = ax.imshow(JJT_conv , cmap='viridis')
            fig.colorbar(im,  orientation='horizontal')
            plt.show()

            # fig.suptitle('NGD Kernel')
            if(1==2):
                bc = BATCH_SIZE * num_classes
                for i in range(6):
                    c = i * 2
                    fig, axs = plt.subplots(1, 2)
                    for row in range(2):
                        ax = axs[row]
                        data = L[row + c][0].reshape(bc, bc)
                        ax.set_title(L[row + c][1])
                        pcm = ax.imshow(data, cmap='viridis')
                        fig.colorbar(pcm,  ax=ax)
                    plt.show()
      
        ###### applying one step for optimization
        optimizer.zero_grad()
        loss = loss_function_none(outputs, sampled_y)
        loss = torch.sum(loss * v_sc)
        loss.backward()

        # last part of SMW formula
        grad_new = []
        for name, param in model.named_parameters():
            param.grad = grad_dict[name] / DAMPING -  param.grad
            # param.grad = grad_dict[name] 
            grad_new.append(param.grad.reshape(1, -1))
        grad_new = torch.cat(grad_new, 1)   
        optimizer.step()
        

        gp = torch.sum( -grad_new * grad_org)
        x = (vjp.unsqueeze(1) -  torch.matmul(NGD_kernel, v) )/ math.sqrt(BATCH_SIZE)
        x = x / DAMPING
        pBp = 0.5 * torch.sum(x * x)
        taylor_appx = loss_org + STEP_SIZE *  gp + STEP_SIZE * STEP_SIZE * pBp
        # taylor_appx = loss_org + gp + pBp
        eps = 0.25
        if batch_idx > 0 or epoch > 0:
            ro =  (loss_org - loss_prev)/ (loss_org - taylor_appx_prev)
            # print(ro)
            if ro > eps:
                alpha_lm = alpha_lm * 0.99
            else:
                alpha_lm = alpha_lm * 1.01
        #     # print(ro)
        loss_prev = loss_org
        taylor_appx_prev = taylor_appx
        # print(descent)

        # print(get_diff(grad_new, grad_org))
        # if batch_idx > 100:
        #     break
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


