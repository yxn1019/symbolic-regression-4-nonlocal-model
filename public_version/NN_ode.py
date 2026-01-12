import numpy as np
import torch
from torch.nn import Linear,Tanh,Sequential
# import torch.nn.functional as F
# from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import random
# import torch.nn.functional as func
import os
# from matplotlib import cm
from scipy.interpolate import interp1d
import scipy.io as io
#%% -------------------set printoptions-------------------------
torch.set_printoptions(precision=7, threshold=None, edgeitems=None, linewidth=None, profile=None)

#------------------Neural network setting-----------------
#Loss_function
# class WeightedPINNLossFunc(nn.Module):
#     def __init__(self, h_data, weight_factor=10.0):
#         super(WeightedPINNLossFunc, self).__init__()
#         self.h_data = h_data
#         self.weight_factor = weight_factor

#     def forward(self, prediction):
#         # 对低值数据赋予更高的权重
#         weight = torch.where(self.h_data < 0.1, self.weight_factor, 1.0)
#         f1 = torch.log(1 + torch.pow((prediction - self.h_data), 2))
#         weighted_loss = (weight * f1).sum()
#         MSE = weighted_loss / total
#         return MSE

class PINNLossFunc(nn.Module):
    def __init__(self, h_data):
        super(PINNLossFunc, self).__init__()
        self.h_data = h_data
        self.use_l2 = True  # 初始阶段使用L2损失

    def set_phase(self, use_l2):
        """设置损失阶段：True为L2，False为对数损失"""
        self.use_l2 = use_l2

    def forward(self, prediction):
        squared_error = torch.pow((prediction - self.h_data), 2)
        if self.use_l2:
            loss = squared_error.mean()
        else:
            loss = torch.log(1 + squared_error).mean()
        return loss
    
class SinActivation(torch.nn.Module):
    def forward(self, x):
        return torch.sin(x)
class GaussianActivation(torch.nn.Module):
    def forward(self, x):
        return torch.exp(-x ** 2)
    
class LearnableGaussianActivation(nn.Module):
    def __init__(self):
        super(LearnableGaussianActivation, self).__init__()
        # Initialize learnable parameters mu and sigma
        self.mu = nn.Parameter(torch.randn(1))  # Mean parameter
        self.sigma = nn.Parameter(torch.ones(1))  # Standard deviation parameter

    def forward(self, x):
        # Gaussian function: exp(-(x - mu)^2 / (2 * sigma^2))
        return torch.exp(-((x - self.mu) ** 2) / (2 * self.sigma ** 2))
class LowValueActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x) + torch.exp(-x ** 2)

##### Neural Network

# Net=Sequential(
#     Linear(1, 20),
#     Tanh(),
#     Linear(20, 20),
#     Tanh(),
#     Linear(20, 20),
#     Tanh(),
#     Linear(20, 20),
#     Tanh(),
#     Linear(20, 20),
#     Tanh(),
#     Linear(20, 20),
#     Tanh(),
#     Linear(20, 20),
#     Tanh(),
#     Linear(20, 20),
#     Tanh(),
#     Linear(20, 1),
# )

Net = Sequential(
    Linear(1, 20),
    LearnableGaussianActivation(),  # Custom Gaussian activation
    Linear(20, 20),
    LearnableGaussianActivation(),
    Linear(20, 20),
    LearnableGaussianActivation(),
    Linear(20, 20),
    LearnableGaussianActivation(),
    Linear(20, 20),
    LearnableGaussianActivation(),
    Linear(20, 1)
)

os.chdir('d:\onedrive - hhu\ml codes\laplace-sr')
# Max_iter_num=2000000
Max_iter_num=15001
noise_level = 0
filename = 'river014'
dataset = io.loadmat(f'dataset/{filename}.mat')
un = np.real(dataset['y'])
un_noise = un
# un_noise = un*(1+noise_level/100*np.random.uniform(-1, 1, size=un.shape))
# un_noise = np.log(un_noise).reshape(-1,1)
# un_noise = un_noise + 20
t = np.real(dataset['t']).reshape(-1,1)
# t = t.astype(np.float32)
t = t.astype(np.float32)
t  = t - 42
# t = np.log(t)

x_num=1
t_num=len(t)
total=x_num*t_num   #Num of total data
choose=int(len(un)*0.7)  #Num of training data
choose_validate=t_num-choose  #Num of validate data   

# Optimizer
# optimizer=torch.optim.Adam([
#     {'params': Net.parameters()}
#     #{'params': theta},
# ], lr=1e-3, weight_decay=1e-5)
optimizer = torch.optim.Adam(Net.parameters(), lr=1e-3, weight_decay=1e-4)
#---------------Create Folder----------------------
try:
    os.makedirs('random_ab')
except OSError:
    pass

try:
    os.makedirs(f'model_save/{filename}-%d-%d'%(choose,noise_level))
except OSError:
    pass

#%-----------------Preparing Training and Validate Dataset
un_raw=torch.from_numpy(un_noise.astype(np.float32))
data=torch.zeros(1)
h_data=torch.zeros([total,1])
database=torch.zeros([total,1])
num=0

    # for j in range(x_num):
        # data[0]=x[j]
data=torch.tensor(t,requires_grad=True).float()
h_data=torch.tensor(un_noise) #Add noise
database=data

# plt.plot(t,h_data,'o')
# plt.show()
#%%-----------Randomly choose----------------
# a = random.sample(list(range(0,50))+list(range(60, total)), choose)
a = random.sample(range(0, total), choose)
np.save("random_ab/"+"a-%d.npy"%(choose),a)
temp=[]
for i in range(total):
    if i not in a:
        temp.append(i)
b=random.sample(temp, choose_validate)

h_data_choose = torch.zeros([choose, 1])
database_choose = torch.zeros([choose, 1])
h_data_validate= torch.zeros([choose_validate, 1])
database_validate = torch.zeros([choose_validate, 1])
num = 0
# for i in a:
#     h_data_choose[num] = h_data[i]
#     database_choose[num] = database[i]
#     num += 1
# num=0
h_data_choose = h_data[a]
database_choose = database[a]
# for i in b:
#     h_data_validate[num] = h_data[i]
#     database_validate[num] = database[i]
#     num += 1
h_data_validate = h_data[b]
database_validate = database[b]

torch.manual_seed(525)
with open(f'model_save/{filename}-%d-%d/'%(choose,noise_level)+'data.txt', 'w') as f:  # 设置文件对象
    threshold = int(0.7 * Max_iter_num)
    A = PINNLossFunc(h_data_choose)    
    for i in range(Max_iter_num):
        optimizer.zero_grad()
        prediction = Net(database_choose)
        prediction_validate = Net(database_validate).cpu().data.numpy()
        ### 根据当前迭代次数设置损失阶段
        # if i < threshold:
        #     A.set_phase(use_l2=True)  # 前30%使用L2
        # else:
        #     A.set_phase(use_l2=False)  # 后70%使用对数损失 
        loss = A(prediction)
        loss_validate = np.sum((h_data_validate.data.numpy() - prediction_validate) ** 2) / choose_validate
        # loss_validate = np.sum((h_data_validate.data.numpy() - prediction_validate) ​**​ 2) / choose_validate
        loss.backward(retain_graph=True)
        optimizer.step()
        if i % 1000 == 0:
            print("iter_num: %d      loss: %.8f    loss_validate: %.8f" % (i, loss, loss_validate))
            f.write("iter_num: %d      loss: %.8f    loss_validate: %.8f \r\n" % (i, loss, loss_validate))
            if int(i / 100) == 800:
                # sign=stop(loss_validate_record)
                # if sign==0:
                #     break
                break
            if i>1000:
                torch.save(Net.state_dict(), f'model_save/{filename}-%d-%d/'%(choose,noise_level)+f"{filename}-%d.pkl"%(i))

#%
# t = database_validate[:,1]
# t = t.data.numpy()
# x = x.data.numpy()


# ax = fig.add_subplot(projection='3d')
# X, T = np.meshgrid(x,t) #mesh for train
# ax.plot_surface(T, X, c, cmap='viridis') #NN模拟值
# ax.scatter(T, X, un_noise,               #加噪精确解
#             facecolors = 'none', 
#             marker = '*', 
#             edgecolor = 'k', 
#             s = 30,
#             label = 'Exact')
#%%
c = Net(database).data.numpy().reshape(-1,x_num)
un_denoise = h_data.data.numpy().reshape(-1,x_num)
# exp_t = np.exp(t)
# exp_c = np.exp(c)
# exp_un_noise = np.exp(un_denoise)
fig = plt.figure()
# ax.set_xlabel(r'$T$')
# ax.set_ylabel(r'$X$')
# ax.set_zlabel(r'$u$')
plt.semilogy(t,c)
# plt.plot(t,c)
plt.plot(t,un_noise,'x')
plt.show()