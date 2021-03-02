import torch
import torchvision
# The main BackPACK functionalities
from backpack import backpack, extend
# The diagonal GGN extension
# from backpack.extensions import DiagGGNMC
import torch.optim as optim
from backpack.extensions import TRIAL
from torchsummary import summary
import time

# This layer did not exist in Pytorch 1.0

# Hyperparameters
# 0: matmul
# 1: fft
# 2: conv2d
MODE = 1
BATCH_SIZE = 5
num_classes = 10
STEP_SIZE = 0.01
DAMPING = 1.0
MAX_ITER = 3
torch.manual_seed(0)
bc = BATCH_SIZE * num_classes

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
    torch.nn.Conv2d(1, 1000, 3, 1, padding = (1,1)),
    torch.nn.ReLU(),
    torch.nn.Conv2d(1000, 1, 3, 1, padding = (1,1)),
    torch.nn.Flatten(), 
    torch.nn.Linear(784, 10),
).to(device)

##### fully connected network. Test for linear timings.
# model = torch.nn.Sequential(
#     torch.nn.Flatten(),
#     torch.nn.Linear(784, 1000),
#     torch.nn.ReLU(),
#     # torch.nn.Sigmoid(),
#     torch.nn.Linear(1000, 10),
# ).to(device)

summary(model, ( 1, 28, 28))

loss_function = torch.nn.CrossEntropyLoss()

def get_accuracy(output, targets):
    """Helper function to print the accuracy"""
    predictions = output.argmax(dim=1, keepdim=True).view_as(targets)
    return predictions.eq(targets).float().mean().item()


# class TrialOptimizer(torch.optim.Optimizer):
#     def __init__(self, parameters, step_size, damping):
#         super().__init__(
#             parameters, 
#             dict(step_size=step_size, damping=damping)
#         )

#     def step(self):
#         for group in self.param_groups:
#             for p in group["params"]:
#                 step_direction = p.grad / (p.trial + group["damping"])
#                 p.data.add_(-group["step_size"], step_direction)
#         return loss

extend(model)
extend(loss_function)

# optimizer = TrialOptimizer(
#     model.parameters(), 
#     step_size=STEP_SIZE, 
#     damping=DAMPING
# )

optimizer = optim.SGD(model.parameters(), lr=STEP_SIZE)


def get_diff(A, B):
    ''' returns relative error between A and B
    '''
    return torch.norm(A - B)/torch.norm(A)


def naive_seq():
    jac_list = []
    for j in range(num_classes):
        for i in range(BATCH_SIZE):
            output[i,j].backward(retain_graph=True)
            L = []
            for name, param in model.named_parameters():
                L.append(param.grad.view(1, -1))
                param.grad = None
            jac_list.append(torch.cat(L, 1))
    jac = torch.cat(jac_list, 0)
    JJT = torch.matmul(jac, jac.permute(1,0))/BATCH_SIZE
    return JJT


def naive_vmap():
    I_N = torch.eye(num_classes)
    # torch._C._debug_only_display_vmap_fallback_warnings(True)
    L = []
    def get_jacobian(v):
        j = torch.autograd.grad(output[i,:], model.parameters(), v, retain_graph = True)
        jac_persample = []
        for j_ in j:
            jac_persample.append(j_.view( -1))
        for name, param in model.named_parameters():
            param.grad = None
        return torch.cat(jac_persample, 0)

    for i in range(BATCH_SIZE):
        jacobian = torch.vmap(get_jacobian)(I_N)
        L.append(jacobian)

    jac = torch.cat(L, 0)
    jac = jac.reshape(BATCH_SIZE, num_classes, -1)
    jac = jac.permute(1, 0 , 2)
    jac = jac.reshape(BATCH_SIZE * num_classes, -1)
    JJT = torch.matmul(jac, jac.permute(1,0))/BATCH_SIZE
    return JJT

def optimal_JJT():
    jac_list = 0
    bc = BATCH_SIZE * num_classes
    with backpack(TRIAL(MODE)):
        loss = loss_function(output, y)
        loss.backward(retain_graph=True)
    for name, param in model.named_parameters():
        trial_vals = param.trial
        jac_list += trial_vals.reshape(bc, bc)
        param.grad = None
    JJT = jac_list / BATCH_SIZE
    return JJT

for batch_idx, (x, y) in enumerate(mnist_loader):
    print('*' * 30)
    output = model(x)
    accuracy = get_accuracy(output, y)

    ######## calling individual function for JJT computation
    ### Our extension
    start_time = time.time()
    JJT_opt = optimal_JJT()
    time_opt = time.time() - start_time
    # print(JJT_opt)

    ### naive loop which is current PyTorch approach
    start_time = time.time()
    JJT_naive_seq = naive_seq()
    time_seq = time.time() - start_time
    # print(JJT_naive_seq)

    ### vamp is slow and not worth it
    # start_time = time.time()
    # JJT_naive_vmap = naive_vmap()
    # time_vmap = time.time() - start_time
    
    



    # loss = model_fn_normal()
    # loss, jac =  model_fn_ngd()
    # jac_list = torch.matmul(jac, jac.permute(1,0))/BATCH_SIZE
    # print(JJT)

    # loss, trial_vals, jac_list = model_fn_trial()
    # v= 10
    # jac_list = jac_list.reshape(BATCH_SIZE*v,BATCH_SIZE*v)/BATCH_SIZE
    # print(jac_list.shape)
    # jac_list = jac_list.reshape(bc, bc)/BATCH_SIZE
    # print(jac_list)

    # applying one step for optimization
    loss = loss_function(output, y)
    loss.backward()
    optimizer.step()

    # print('Seq vs vmap error:', get_diff(JJT_naive_seq, JJT_naive_vmap))
    print('opt vs seq error:', get_diff(JJT_naive_seq, JJT_opt))
    print('Jacobian Computation Time [Sequential]:', time_seq)
    print('Jacobian Computation Time [Optimal]:', time_opt)
    # print('Jacobian Computation Time [VMAP]:', time_vmap)
    print(
        "Iteration %3.d/%d   " % (batch_idx, MAX_ITER) +
        "Minibatch Loss %.3f  " % (loss) +
        "Accuracy %.0f" % (accuracy * 100) + "%"
    )

    if batch_idx >= MAX_ITER:
        break

