# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 10:55:14 2024

@author: Xiangnan Yu

This example will show the detailed discovery process of 
memory function of Application case 3

"""
import os
import numpy as np
from scipy.special import roots_laguerre
import matplotlib.pyplot as plt
# import sympy as sp
import scipy.io as io
from scipy.integrate import quad
import re
# import sys
import torch
import torch.nn as nn
from torch.nn import Linear,Tanh,Sequential
# import torch.nn.functional as F'
# from torch.autograd import Variable
# from pysr import PySRRegressor
from scipy.interpolate import interp1d

class LearnableGaussianActivation(nn.Module):
    def __init__(self):
        super(LearnableGaussianActivation, self).__init__()
        # Initialize learnable parameters mu and sigma
        self.mu = nn.Parameter(torch.randn(1))  # Mean parameter
        self.sigma = nn.Parameter(torch.ones(1))  # Standard deviation parameter
    def forward(self, x):
        # Gaussian function: exp(-(x - mu)^2 / (2 * sigma^2))
        return torch.exp(-((x - self.mu) ** 2) / (2 * self.sigma ** 2))

def ft_nn(t):
    # print(t)
    t = torch.tensor(t, dtype=torch.float32).reshape(-1,1)
    # t = torch.log(t)
    ft = Net(t).detach().numpy()
    # ft = np.exp(ft-20)     
    # print(t.numpy(),'\t',ft)
    return ft    
    
# def laplace_transform_quad(f, s):
#     integral = np.zeros(len(s))
#     for i in range(len(s)):
#         integral[i], _ = quad(lambda t: np.exp(-s[i] * t) * f(t), 0, np.inf)
#     return integral
    # integral, error = quad(lambda t: np.exp(-s * t) * f(t), 0, np.inf)
    # return integral
def laplace_transform_gaussian_laguerre(f, s, n):
    roots, weights = roots_laguerre(n)
    f_scale = np.array([f(r / s) for r in roots]).reshape(n,-1)
    laplace_approx = np.zeros([len(s), 1])   
    # print(f_scale.shape, weights[:, np.newaxis].shape, s.shape)
    laplace_approx = np.sum(f_scale * weights[:, np.newaxis], axis=0)/s
    # print(f_scale.shape, weights[:, np.newaxis].shape)
    return laplace_approx

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
# acquire the current path
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
# change the workspace to the current path
os.chdir(current_dir)
# os.chdir('d:\onedrive - hhu\ml codes\laplace-sr')
noise_level = 0
filename = 'river051'  #
L = re.findall(r'-?\d+\.?\d*', filename)
if L==['10']: L = ['100']
L = np.array(L).astype(float)/10
data = io.loadmat(f'dataset/{filename}.mat')
u_data = np.real(data['y']).reshape(-1,1)

# u_data_noise=u_data*(1+noise_level/100*np.random.uniform(-1, 1, size=u_data.shape))
# x = np.real(data['x'][0])
t_data = np.real(data['t']).reshape(-1,1)
choose = int(len(t_data)*0.7)  #训练集分割
# u_data[u_data < 1e-5] = 0
# Net.load_state_dict(torch.load(f'model_save/{filename}-{choose}-{noise_level}/{filename}-15000.pkl'))
# sys.path.append(path)
# os.chdir(os.path.dirname(path))
# data = io.loadmat(path+'dataset/tsfade_fft.mat')
# un = np.real(data['Exact'])
# # x = np.real(data['x'][0])
# t = np.real(data['t']).flatten()
# x = np.real(data['x']).flatten()


##########
#---------- step 1: parameter setup & data input-----------
##########

'''
The Red Cedar River experiments involve three distinct subcases based on the 
measurement distance L:
    1. L = 1.4 km
    2. L = 3.1 km
    3. L = 5.1 km

The number of Gaussian quadrature nodes (n), the magnitude of the time shift, 
and the range of the Laplace variable (s) vary across these subcases. 
Specifically:

    (1) Influence of Quadrature Nodes (n):
        Empirically, the number of quadrature nodes has a negligible effect on 
        the Laplace transform  as long as n > 10. However, we selected a 
        sufficiently large n to ensure it covers the total measurement duration 
        for each location (L).

    (2) Time Shift:
        A time shift is applied to facilitate the Laplace transform, as the 
        input data must strictly start at t = 0. The datasets are shifted in 
        the time domain and subsequently corrected using the time-shifting 
        property of the Laplace transform: L{f(t-shift)}=exp(-s*shift)*F(s).

    (3) Scope of Laplace Variable (s):
        The range of s is determined by the numerical Laplace transform
        algorithm, specifically defined as [6/max_time, 6/min_time], where 6 
        is a characteristic coefficient of the algorithm. Note: For the case
        where L = 1.4 km, the 'min_time' parameter used does not reflect the 
        actual minimum measurement time. This adjustment was necessary because
        the transformed data f(s) derived from the true minimum was unusual
        and physically uninterpretable, likely due to measurement errors at 
        late time.
'''
n_s = 100  # number of laplace variables s
if L == 5.1:
    n = 35 # number of gaussian-quadrature nodes
    shift = 130 # length of shift
    s = np.linspace(6/270, 6/0.01, n_s).reshape(-1,) # scope of Laplace transform
elif L == 3.1:
    n = 22     # Number of quadrature points
    shift = 90
    s = np.linspace(6/200, 6/90, n_s).reshape(-1,)
elif L== 1.4:
    u_data[0] = False
    n = 16
    shift = 42
    s = np.linspace(6/100, 6/6, n_s).reshape(-1,)
else:
    n = 50
    shift = 0
    s = np.linspace(6/t_data[-1], 6/t_data[0], n_s).reshape(-1,)
    print('You choose not to process the Cedar River data')

t_data = t_data.reshape(-1,) - shift 
u_data = u_data.reshape(-1,)
# u_data = u_data/sum(u_data)


# if L == 3.1: 
#     u_data[-1] = u_data[-1] - 0.45
#     t_data = np.hstack(([0.],t_data))
#     u_data = np.hstack(([0.],u_data))
# if L == 1.4:
    # u_data[0:2] = u_data[0:2] + 
t, w = roots_laguerre(n)
# t = np.tensor(t, dtype=torch.float32)
w = np.array(w, dtype=np.float32)
# s = np.array(s, dtype=np.float32)
t_scale = t

ft_spline = interp1d(t_data, u_data, kind='linear', fill_value="extrapolate") 
f_scale = ft_spline(t_scale) # '''spline interpolation'''
# f_scale = ft_min(t)
# f_scale = ft_nn(t)

###########
#-------------step 2: numerical Laplace transform-------------
###########
fs = laplace_transform_gaussian_laguerre(ft_spline, s, n)
# fs = laplace_transform_gaussian_laguerre(ft_nn, s, n)
# fs = laplace_transform_gaussian_laguerre(ft_nn, s, n)

plt.figure(1)
# plt.plot(t_data,u_data,'o', label='Raw data')
plt.semilogy(t_data,u_data,'o', label='Raw data')
plt.plot(t_scale,f_scale, label='Reconstruced data')
plt.xlabel('Time (h)')
plt.ylabel('Concentration')
# plt.legend()
plt.show() 
#%
plt.figure(2)
# fs=fs*np.exp(-shift*s)
ks = -1/L*np.log(fs)
# ks = -1/L*np.log(np.exp(-shift*s)*fs) 
# s = np.hstack((0., s))
# ks = np.hstack((0., ks))
# ke = -1/L*np.log(np.exp(-shift*s)/sum(u_data)) 

plt.plot(s,ks,'+', label='Transformed data')
# plt.plot(s,ke)
# plt.loglog(s,ks,'+', label='Transformed data')
# ks_recover =  s**0.23226008/np.exp(4.4498343/s)
# plt.loglog(s
# #             # 1000*(1-s**0.45)
# #             fs_true
#               # np.exp(s**0.8969641/(-0.48029226))/s
#              , ks_recover
#             , label='Learned expression'
#             )
plt.xlabel('s')
plt.ylabel(r'$\~\mathcal{K}(s)$')
plt.figure(3)
plt.loglog(s,fs.reshape(-1,1))
# plt.plot(s,fs.reshape(-1,1))
# plt.loglog(s,np.exp(-shift*s).reshape(-1,1),'o')
# plt.xlabel('s')
# plt.ylabel(r'$exp(shift s)*f(s)$')

# plt.figure(4)
# plt.loglog(s,fs,'o')
plt.xlabel('s')
plt.ylabel(r'$f(s)$')
# plt.legend()
plt.show()
 
#%%
y = ks ####################y轴数据
# y = fs
x0 = s.reshape(-1,1)   ####################x轴
seg_loss1 = """
function sequential_segmented_loss(tree, dataset::Dataset{T,L}, options; 
                                   epsilon=1e-6,
                                   weight_power=1.0)::L where {T,L}  # 新增权重指数参数
    y_pred, completed = eval_tree_array(tree, dataset.X, options)
    !completed && return L(Inf)
    
    # 按顺序分割索引
    n_samples = length(dataset.y)
    
    # 加权损失
    weighted_part = let
        # 提取后半段数据
        y_true_seg = abs.(dataset.y) .+ epsilon
        y_pred_seg = abs.(y_pred) .+ epsilon
        
        # 计算权重：小值获得更高权重，权重 = 1/(y_true^power + epsilon)
        weights = 1.0 ./ (y_true_seg .^ weight_power .+ epsilon)  # 可调节的权重计算
        
        # 计算加权绝对误差（可改为平方误差）
        # weighted_errors = weights .* abs.(y_pred_seg .- y_true_seg)
        
        # 平方加权误差（根据需求选择）
         weighted_errors = weights .* (y_pred_seg .- y_true_seg).^2
        
        sum(weighted_errors)
    end
    
    # 组合并归一化（根据误差类型调整）
    return weighted_part / n_samples
end
"""
seg_loss2 ="""
function sequential_segmented_loss(tree, dataset::Dataset{T,L}, options; 
                                   split_ratio1=0.4,
                                   split_ratio2=0.4,
                                   epsilon=1e-6)::L where {T,L}
    y_pred, completed = eval_tree_array(tree, dataset.X, options)
    !completed && return L(Inf)
    
    # Split index of data in order
    n_samples = length(dataset.y)
    split_idx1 = floor(Int, split_ratio1 * n_samples)
    split_idx2 = floor(Int, split_ratio2 * n_samples)
    
    # Calculate the L2 loss for the first half
    log1_part = let
        y_true1 = abs.(dataset.y[1:split_idx1]) .+ epsilon
        y_pred_seg1 = abs.(y_pred[1:split_idx1]) .+ epsilon
        sum((log.(y_pred_seg1) .- log.(y_true1)).^2)
    end
    # Calculate the L2 loss for the second half
    l2_part = sum((y_pred[split_idx1+1:split_idx2] .- dataset.y[split_idx1+1:split_idx2]).^2)
    
    # Calculate the log-L2 loss for the third half
    log2_part = let
        y_true2 = abs.(dataset.y[split_idx2+1:end]) .+ epsilon
        y_pred_seg2 = abs.(y_pred[split_idx2+1:end]) .+ epsilon
        sum((log.(y_pred_seg2) .- log.(y_true2)).^2)
    end
    
    # combination of the three part
    return (log1_part + l2_part + log2_part) / n_samples
end
"""
from pysr import PySRRegressor
# y = f_scale.reshape(-1,1)
# x0 = t_scale.reshape(-1,1)
model = PySRRegressor(
    model_selection="best",  # Result is mix of simplicity + accuracy
    maxsize = 10,
    niterations = 2000,
    binary_operators=[ "*", "/", "^"
                      , "+"
                      ],
    unary_operators=[       
        "exp",
        "log",
        # "inv(x) = 1/x",
    #     # ^ Custom operator (julia syntax)
    ],
    # constraints={"pow": (-2, 2)},
    # complexity_of_constants = 0,
    # complexity_of_operators={
    #                         # "^": 0, 
    #                          # "exp": 0,
    #                          'log': 2,
                             
    #                          },
    nested_constraints={
        "exp": {"exp": 0, "^": 0, "log": 0}, 
        # "exp": {"^": 1}, #{"sin": {"cos": 0}}, "cos": {"cos": 2}}
        "^": { "exp": 0, "^": 0, "log": 0},
        "log": {"exp": 0, "^": 0, "log": 0}, 
        },
    # extra_sympy_mappings={"inv": lambda x: 1 / x},
    # ^ Define operator for SymPy as well
    elementwise_loss="loss(x, y) = (x - y)^2",
    # elementwise_loss="loss(x, y) = (x - y)^2/(abs(y)+1e-7)",
    # loss_function=seg_loss2,
    # ^ Custom loss function (julia syntax)
)

model.fit(x0, y)
print(model)

print( '\n\n Sym reg:', model.sympy())

# %% print results
'''the best model in our manuscript has been stored as below:'''
# model = PySRRegressor.from_file(
#     # run_directory='outputs/20250917_204251_IX9iEM' # 1.4 l2_loss paper
#     run_directory='outputs/20250815_170117_jwFhjk' # 3.1 fit fs b=0   
#     # run_directory='outputs/20250809_153738_omsiS5' # 5.1 fit fs, b = 0
#     # run_directory='outputs/20250921_133133_n0dWsq' # saa10, using stm for mass tranfer
    
#     ####### Alternative candidates:
#     # run_directory='outputs/20250814_172657_YmPzMV' # 1.4, b = 3 fit fs
#     # run_directory='outputs/20251209_162733_QQ3LyB' # 5.1 b = 0 paper
#     # run_directory='outputs/20250917_211023_cKBbhY' # 3.1 paper
#     )
b = 0
if b == 0:
    print('\n Sym reg:', model.sympy())
    f = model.predict(x0,)
else:
    print('\n Sym reg:', model.sympy(b))
    f = model.predict(x0,b)
# plt.plot(x0,y,'o')
# plt.plot(x0,f)
plt.figure()
plt.plot(x0,y,'o', label='Data')
# plt.loglog(x0,y,'o', label='Data')
plt.plot(x0,f, label='Recovered model')
# plt.plot(x0,f)
# 设置 y 轴范围（示例：限制在 [y_min, y_max]）
# y_min = 1e-2  # 最小 y 值（根据数据调整）
# y_max = 1e1   # 最大 y 值（根据数据调整）
# plt.ylim(y_min, y_max)  # 关键：限制 y 轴范围
plt.xlabel('s')
plt.ylabel('$\mathcal{K}$(s)')
plt.legend()
plt.show()

