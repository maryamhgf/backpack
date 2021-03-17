import torch
import torchvision
from backpack import backpack, extend
import torch.optim as optim
from backpack.extensions import Fisher, BatchGrad
from torchsummary import summary
import time

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
DAMPING = 0.1
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
# model = torch.nn.Sequential(
#     torch.nn.Conv2d(1, 50, 3, 1, padding = (1,1)),
#     # torch.nn.BatchNorm2d(50),
#     torch.nn.ReLU(),

#     torch.nn.Conv2d(50, 50, 3, 1, padding = (1,1)),
#     # torch.nn.BatchNorm2d(50),
#     torch.nn.ReLU(),

#     torch.nn.Conv2d(50, 10, 3, 1, padding = (1,1)),
#     # torch.nn.BatchNorm2d(10),
#     torch.nn.ReLU(),

#     torch.nn.Flatten(), 
#     torch.nn.Linear(28*28*10, 20),
#     # torch.nn.BatchNorm1d(20),

#     torch.nn.ReLU(),

#     torch.nn.Linear(20, 100),
#     # torch.nn.BatchNorm1d(100),
#     torch.nn.ReLU(),

#     torch.nn.Linear(100, 10),


# ).to(device)



##### fully connected network. Test for linear timings.
model = torch.nn.Sequential(
    torch.nn.Flatten(), 
    torch.nn.Linear(28*28, 500),
    torch.nn.BatchNorm1d(500),
    torch.nn.ReLU(),
    torch.nn.Linear(500, 500),
    torch.nn.ReLU(),
    torch.nn.Linear(500, 500),
    torch.nn.ReLU(),
    torch.nn.Linear(500, 500),
    torch.nn.ReLU(),
    torch.nn.Linear(500, 500),
    torch.nn.ReLU(),
    torch.nn.Linear(500, 10)
).to(device)

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


# def backpack_batch_grad():
#     jac_list = 0
#     with backpack(BatchGrad()):
#         loss = loss_function(output, y)
#         loss.backward(retain_graph=True)
#     for name, param in model.named_parameters():
#         # multiple by batch size to get the original gradient
#         all_grad = BATCH_SIZE * param.grad_batch.reshape(BATCH_SIZE, -1)
#         jac_list += torch.matmul(all_grad, all_grad.t())
#         param.grad_batch = None
#         param.grad = None
#     JJT = jac_list / BATCH_SIZE

#     return JJT

def optimal_JJT(acc_test=False):
    jac_list = 0
    batch_grad_list = 0
    if acc_test:
        with backpack(Fisher(), BatchGrad()):
            loss = loss_function(output, y)
            loss.backward(retain_graph=True)
    else:
        with backpack(Fisher()):
            loss = loss_function(output, y)
            loss.backward(retain_graph=True)

    for name, param in model.named_parameters():
        fisher_vals = param.fisher
        jac_list += fisher_vals

        if acc_test:
            all_grad = BATCH_SIZE * param.grad_batch.reshape(BATCH_SIZE, -1)
            batch_grad_list += torch.matmul(all_grad, all_grad.t())
            param.grad_batch = None

        param.fisher = None
        param.grad = None

    JJT_backpack = batch_grad_list / BATCH_SIZE
    JJT = jac_list / BATCH_SIZE

    if acc_test:
        print('Estimation Error:', get_diff(JJT_backpack, JJT))

    return JJT

def optimal_JJT_blk():
    jac_list = 0
    bc = BATCH_SIZE * num_classes
    # L = []

    with backpack(TRIAL(MODE)):
        loss = loss_function(output, y)
        loss.backward(retain_graph=True)
    for name, param in model.named_parameters():
        trial_vals = param.trial
        # L.append([trial_vals / BATCH_SIZE, name])
        jac_list += torch.block_diag(*trial_vals)
        param.trial = None
    JJT = jac_list / BATCH_SIZE
    return JJT

acc_list = []
time_list = []
loss_list = []
epoch_time_list = []
start_time= time.time()
for epoch in range(EPOCHS):
    start_time_epoch = time.time()
    for batch_idx, (x, y) in enumerate(mnist_loader):
        # y, indices = torch.sort(y)
        # x = x[indices, :, :, :]
        x, y = x.to(device), y.to(device)
        output = model(x)
        accuracy = get_accuracy(output, y)

        ######## calling individual function for JJT computation
        ### Our extension
        
        JJT_opt = optimal_JJT(acc_test=False)
        NGD_kernel = JJT_opt
        v_mat = torch.linalg.inv(NGD_kernel + DAMPING * torch.eye(BATCH_SIZE))
        v = torch.sum(v_mat, dim=0)/BATCH_SIZE
        # print(JJT_opt)

        # print('linear + diag:', get_diff(JJT_opt, JJT_linear + JJT_conv * torch.eye(BATCH_SIZE)))
        # print('linear:', get_diff(JJT_opt, JJT_linear))
        # print('^^^^^^^^^^^^^^')
        # x = torch.ones(1, BATCH_SIZE, BATCH_SIZE)
        # x = x.repeat(num_classes, 1, 1)
        # eye_blk = torch.block_diag(*x)
        # JJT_opt_blk = JJT_opt * eye_blk
        # JJT_conv_blk = JJT_conv * eye_blk
        # JJT_fused = JJT_conv_blk + JJT_linear

        ### Blocked NGD version
        # start_time = time.time()
        # JJT_opt_blk = optimal_JJT_blk()
        # print(torch.norm(JJT_opt))
        # print(JJT_opt)
        # time_opt = time.time() - start_time

        # plotting NGD kernel for some iterations
        if PLOT and batch_idx in [2, 10, 50, 500] :
            # JJT_opt_blk = optimal_JJT_blk()

            JJT_opt, JJT_linear, JJT_conv = optimal_JJT()
            # x = torch.ones(1, BATCH_SIZE, BATCH_SIZE)
            # x = x.repeat(num_classes, 1, 1)
            # eye_blk = torch.block_diag(*x)
            # diff = JJT_opt - JJT_opt*eye_blk
            # u, s, vh = torch.linalg.svd(diff)
            # s_normal = torch.cumsum(s, dim = 0)/torch.sum(s)
            # print(s_normal.numpy())
            # fig, ax = plt.subplots()
            # im = ax.plot(s_normal)
            # print(s)
            # fig.colorbar(im,  orientation='horizontal')
            # plt.show()
            
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
                        print('name:', L[row + c][1])
                        print('max data:', torch.max(data))
                        print('min data:', torch.min(data))
                        print('average data:', torch.mean(data))
                        print('norm data:', torch.norm(data))

                        ax.set_title(L[row + c][1])
                        pcm = ax.imshow(data, cmap='viridis')
                        fig.colorbar(pcm,  ax=ax)
                    plt.show()
        

        ### backpack original batch grad
        # start_time = time.time()
        # JJT_backpack = backpack_batch_grad()
        # print(JJT_backpack)
        # time_vmap = time.time() - start_time
      
        # applying one step for optimization
        # loss = loss_function(output, y)
        loss = loss_function_none(output, y)
        loss = torch.sum(loss * v)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        

        

        if batch_idx % 1 == 0:
            acc_list.append(accuracy)
            time_list.append(time.time() - start_time)
            loss_list.append(loss)
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
                "Minibatch Loss %.3f  " % (loss) +
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


