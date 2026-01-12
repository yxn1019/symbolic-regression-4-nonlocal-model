# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 10:55:14 2024

@author: Xiangnan Yu

Only datasets and stored results of PySr presented in this manuscript are provided for 
reference. For detailed parameter configurations and additional symbolic 
regression rules, please see example_ctrw/stm.py.
"""
import os
import numpy as np
from scipy.special import roots_laguerre
# import matplotlib
# matplotlib.use('Qt5Agg')  # 使用Qt5后端 
import matplotlib.pyplot as plt
# import sympy as sp
import scipy.io as io
from scipy.integrate import quad
import sys
import torch
from torch.nn import Linear,Tanh,Sequential
# import torch.nn.functional as F'
# from torch.autograd import Variable
# from pysr import PySRRegressor
from scipy.interpolate import interp1d


seg_loss2 ="""
function sequential_segmented_loss(tree, dataset::Dataset{T,L}, options; 
                                   split_ratio1=0.4,
                                   split_ratio2=0.4,
                                   epsilon=1e-6)::L where {T,L}
    y_pred, completed = eval_tree_array(tree, dataset.X, options)
    !completed && return L(Inf)
    
    # 按顺序分割索引
    n_samples = length(dataset.y)
    split_idx1 = floor(Int, split_ratio1 * n_samples)
    split_idx2 = floor(Int, split_ratio2 * n_samples)
    
    # 前半段计算L2损失
    log1_part = let
        y_true1 = abs.(dataset.y[1:split_idx1]) .+ epsilon
        y_pred_seg1 = abs.(y_pred[1:split_idx1]) .+ epsilon
        sum((log.(y_pred_seg1) .- log.(y_true1)).^2)
    end
    #中段
    l2_part = sum((y_pred[split_idx1+1:split_idx2] .- dataset.y[split_idx1+1:split_idx2]).^2)
    
    # 后半段计算对数空间MSE
    log2_part = let
        y_true2 = abs.(dataset.y[split_idx2+1:end]) .+ epsilon
        y_pred_seg2 = abs.(y_pred[split_idx2+1:end]) .+ epsilon
        sum((log.(y_pred_seg2) .- log.(y_true2)).^2)
    end
    
    # 组合并归一化
    return (log1_part + l2_part + log2_part) / n_samples
end
"""

def laplace_transform_gaussian_laguerre(f, s, n):
    roots, weights = roots_laguerre(n)
    f_scale = np.array([f(r / s) for r in roots]).reshape(n,-1)
    laplace_approx = np.zeros([len(s), 1])   
    # print(f_scale.shape, weights[:, np.newaxis].shape, s.shape)
    laplace_approx = np.sum(f_scale * weights[:, np.newaxis], axis=0)/s
    # print(f_scale.shape, weights[:, np.newaxis].shape)
    return laplace_approx

def ft(t):
    # print(t)
    t = torch.tensor(t, dtype=torch.float32).reshape(-1,1)
    t = torch.log(t)
    ft = Net(t).detach().numpy()
    ft = np.exp(ft-20)     
    # print(t.numpy(),'\t',ft)
    return ft

def ft_no(t): #  no transform
    # print(t)
    t = torch.tensor(t, dtype=torch.float32).reshape(-1,1)
    # t = torch.log(t)
    ft = Net(t).detach().numpy()
    # ft = np.exp(ft-20)     
    # print(t.numpy(),'\t',ft)
    return ft    
    
def laplace_transform_quad(f, s):
    # integral = np.zeros(len(s))
    # for i in range(len(s)):
    #     integral[i], _ = quad(lambda t: np.exp(-s[i] * t) * f(t,x), 0, np.inf)
    integral, err = quad(lambda t: np.exp(-s * t) * f(t), 0, np.inf)
    # print(err)
    return integral

Net = Sequential(
    Linear(1, 20),
    Tanh(),
    Linear(20, 20),
    Tanh(),
    Linear(20, 20),
    Tanh(),
    Linear(20, 20),
    Tanh(),
    Linear(20, 20),
    Tanh(),
    Linear(20, 20),
    Tanh(),
    Linear(20, 20),
    Tanh(),
    Linear(20, 20),
    Tanh(),
    Linear(20, 1),
)

####################
#---------- step 1: parameter setup & data input-----------
####################

# acquire the current path
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
# change the workspace to the current path
os.chdir(current_dir)
# path = 'D:/OneDrive - HHU/ML codes/DL-PDE/DL-FDE (code)/'
noise_level = 0
filename = 'tfde'
choose = 70
data = io.loadmat(f'dataset/{filename}.mat')
u_data = np.real(data['y']).reshape(-1,1)
u_data_noise=u_data*(1+noise_level/100*np.random.uniform(-1, 1, size=u_data.shape))
# x = np.real(data['x'][0])
t_data = np.real(data['t']).reshape(-1,1)
# u_data[u_data < 1e-5] = 0
Net.load_state_dict(torch.load(f'model_save/{filename}-{choose}-{noise_level}/{filename}-10000.pkl'))
# sys.path.append(path)
# os.chdir(os.path.dirname(path))

# Parameters

# s = 2
n_s = 100  # number of laplace variable
n = 15     # Number of quadrature nodes
n_x = 10
shift = 0
# s = torch.logspace(np.log10(10), np.log10(100), n_s)   
s = np.linspace(0.001, 1, n_s) # Laplace transform variable
t, w = roots_laguerre(n)
# t = np.tensor(t, dtype=torch.float32)
w = np.array(w, dtype=np.float32)
# s = np.array(s, dtype=np.float32)

num = 0
t_data = t_data.reshape(-1,) - shift
u_data = u_data.reshape(-1,)

# fs_true = np.exp(-s**(0.45)/0.1)/s
t_interp = t
ft_interp = interp1d(t_data, u_data, kind='linear', fill_value="extrapolate") 
ft_scale = ft_interp(t_interp)
fs = laplace_transform_gaussian_laguerre(ft_interp, s, n)

# t_scale = t
# ft_scale = ft_no(t_scale)
# fs = laplace_transform_gaussian_laguerre(ft_no, s, n)


fs = fs*np.exp(-shift*s)
ks = -np.log(s*fs)
plt.figure()
plt.plot(t_data,u_data_noise,'o', label='Raw data')
plt.plot(t_interp,ft_scale, label='Reconstructed data')
# plt.plot(t_scale,ft_scale, label='Reconstructed data')
# plt.plot(t[0]/s,f_interp(t[0]/s))
# plt.plot(t[0]/s,ft(t[0]/s))
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.legend()
#%
plt.figure()
#
ks_recover = (s**0.44745705) * 9.953289
plt.loglog(s, ks,'+', label='K(s)')
plt.plot(s,ks_recover)
plt.show()

#%%
###########
#-------------step 2: numerical Laplace transform-------------
###########
x = s.reshape(-1,1)
y = ks.reshape(-1,1)
from pysr import PySRRegressor
model = PySRRegressor(
    model_selection="best",  # Result is mix of simplicity+accuracy
    niterations=1000,
    maxsize = 5,
    binary_operators=["^", "*", "/"],
    unary_operators=[
    #     # "cos",
        "exp",
    #     # "sin",
        # "inv(x) = 1/x",
    #     # ^ Custom operator (julia syntax)
    ],
    
    nested_constraints={
        "exp": {"exp": 0, "^": 1}, 
        # "exp": {"^": 1}, #{"sin": {"cos": 0}}, "cos": {"cos": 2}}
        "^": {"^": 0, "exp": 0}
        },
    # constraints={"pow": (-2, 2)},
    # extra_sympy_mappings={"inv": lambda x: 1 / x},
    # ^ Define operator for SymPy as well
    elementwise_loss="loss(x, y) = (x - y)^2/(y+1e-7)",
    # elementwise_loss="loss(x, y) = (x - y)^2",
    # loss_function=seg_loss2,
    # ^ Custom loss function (julia syntax)
)

model.fit(x, y)
print(model,'\n')
# %%
model = PySRRegressor.from_file(
    run_directory="outputs/20250602_172957_27m5gR"
    )
b = 4 # 
if b == 0:
    print('\n Sym reg:', model.sympy())
    f = model.predict(x,)
else:
    print('\n Sym reg:', model.sympy(b))
    f = model.predict(x,b)
# plt.plot(x0,y,'o')
# plt.plot(x0,f)
plt.figure()
# plt.semilogy(x,y,'o', label='Synthetic data')
plt.loglog(x,y,'o', label='Transformed data')

plt.plot(x,f, label='Recovered expression')
# plt.plot(x0,f)
# 设置 y 轴范围（示例：限制在 [y_min, y_max]）
# y_min = 1e-3  # 最小 y 值（根据数据调整）
# y_max = np.inf   # 最大 y 值（根据数据调整）
# plt.ylim(y_min)  # 关键：限制 y 轴范围
plt.xlabel('s')
plt.ylabel('$\mathcal{K}(s)$')
plt.legend()
plt.show()