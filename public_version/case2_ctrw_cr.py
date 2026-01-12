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
import matplotlib.pyplot as plt
# import sympy as sp
import scipy.io as io
from scipy.integrate import quad
# import sys
import torch
from torch.nn import Linear,Tanh,Sequential
# import torch.nn.functional as F'
# from torch.autograd import Variable
# from pysr import PySRRegressor
from scipy.interpolate import interp1d



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

def ft_nn(t):
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
##########
#---------- step 1: parameter setup & data input-----------
##########
# os.chdir('d:\onedrive - hhu\ml codes\laplace-sr\public_version') # folder path
# acquire the current path
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
# change the workspace to the current path
os.chdir(current_dir)

noise_level = 0
filename = 'As'
choose = 24
data = io.loadmat(f'dataset/{filename}.mat')
u_data = np.real(data['y']).reshape(-1,1)
u_data_noise=u_data*(1+noise_level/100*np.random.uniform(-1, 1, size=u_data.shape))
# x = np.real(data['x'][0])
t_data = np.real(data['t']).reshape(-1,1)
# u_data[u_data < 1e-5] = 0
Net.load_state_dict(torch.load(f'model_save/{filename}-{choose}-{noise_level}/{filename}-5000.pkl'))
# sys.path.append(path)
# os.chdir(os.path.dirname(path))
# data = io.loadmat(path+'dataset/tsfade_fft.mat')
# un = np.real(data['Exact'])
# # x = np.real(data['x'][0])
# t = np.real(data['t']).flatten()
# x = np.real(data['x']).flatten()
n_s = 100  # number of laplace variables
n = 15     # Number of quadrature points
# s = torch.logspace(np.log10(10), np.log10(100), n_s)   # Laplace transform variable
s = np.linspace(0.01, 1, n_s)
t, w = roots_laguerre(n)
# t = np.tensor(t, dtype=torch.float32)
w = np.array(w, dtype=np.float32)
# s = np.array(s, dtype=np.float32)

num = 0
t_data = t_data.reshape(-1,)
u_data = u_data.reshape(-1,)

###########
#-------------step 2: numerical Laplace transform-------------
###########

fs_true = np.exp(-s**(0.45)/0.1)/s
# f_interp = interp1d(t_data, u_data, kind='cubic',fill_value="extrapolate")
# f_interp = ft(t)
# for i in range(n_s):
#     t_scale = t/s[i]
#     f_scale[i] = np.matmul(ft(t_scale).reshape(-1,n), w.reshape(-1,1))/s[i]
# #     fs[i] = laplace_transform_quad(ft, s[i])
# fs = laplace_transform_gaussian_laguerre(f_interp, s, n)
t_scale = np.linspace(1, 10, 1000)
f_scale = ft_nn(t_scale)
# f_scale = f_interp(t_scale)
fs = laplace_transform_gaussian_laguerre(ft_nn, s, n)
plt.figure()
plt.plot(t_data/60,u_data,'o', label='Raw data')
plt.plot(t_scale,f_scale, label='Reconstruced data')
# plt.plot(t[0]/s,f_interp(t[0]/s))
# plt.plot(t[0]/s,ft_nn(t[0]/s))
plt.xlabel('Time (h)')
plt.ylabel(r'Normalized Concentration ($c/c_0$)')
plt.legend()
plt.show()
#%
plt.figure()
ks = -0.5*np.log(s.reshape(-1,1)*fs.reshape(-1,1))
plt.plot(s,ks,'+', label='Transformed data')
ks_recover =  s**0.92851156*1.0594178
plt.loglog(s
#             # 1000*(1-s**0.45)
#             fs_true
              # np.exp(s**0.8969641/(-0.48029226))/s
             , ks_recover
            , label='Learned expression'
            )

plt.xlabel('s')
plt.ylabel(r'$\~\mathcal{K}(s)$')
plt.legend()
plt.show()

#################################
#%%
###############
#-----------step 3: symbolic regression for memory function------------
###############
from pysr import PySRRegressor
y = ks
# y = -0.5*np.log((s.reshape(-1,1)*fs.reshape(-1,1)))
x0 = s.reshape(-1,1)
model = PySRRegressor(
    model_selection="best",  # Result is mix of simplicity+accuracy
    niterations=400,
    binary_operators=["pow", "*", "/"],
    unary_operators=[
    #     # "cos",
        "exp",
    #     # "sin",
        # "inv(x) = 1/x",
    #     # ^ Custom operator (julia syntax)
    ],
    constraints={"pow": (-2, 2)},
    complexity_of_constants = 0,
    # extra_sympy_mappings={"inv": lambda x: 1 / x},
    # ^ Define operator for SymPy as well
    elementwise_loss="loss(x, y) = (x - y)^2/(abs(y)+1e-7)",
    # ^ Custom loss function (julia syntax)
)

model.fit(x0, y)
print(model)
print( '\n\n Sym reg:', model.sympy())
###############
#-----------step 4: print results------------
###############
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
plt.loglog(x0,y,'o', label='Synthetic data')
plt.plot(x0,f, label='Recovered model')
# plt.loglog(x0,y,'o')
# plt.plot(x0,f)
# 设置 y 轴范围（示例：限制在 [y_min, y_max]）
# y_min = 1e-2  # min y value
# y_max = 1e1   # max y value
# plt.ylim(y_min, y_max)  # scope of y-axis
plt.xlabel('s')
plt.ylabel('$\mathcal{K}$(s)')
plt.legend()
plt.show()