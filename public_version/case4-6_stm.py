# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 14:38:48 2025

@author: Xiangnan Yu
"""

import os
from pysr import PySRRegressor
import numpy as np
from numpy import exp
# from scipy.special import roots_laguerre, roots_genlaguerre, gamma
# from scipy.integrate import quad
import matplotlib.pyplot as plt
# import sympy as sp
import scipy.io as io
from juliacall import Main as jl
import re
# import sys
# import torch
# from torch.nn import Linear,Tanh,Sequential
# import torch.nn.functional as F
# from torch.autograd import Variable
# %
os.chdir('d:\onedrive - hhu\ml codes\laplace-sr') # personal path
#%%
filename = 'saa2' # saa2, br2, tr2 and sediment01 是用于训练sr的数据集
                 # sediment01包含22 44
                 # saariver051 用于对照实验
data = io.loadmat(f'dataset/{filename}.mat')
p = np.real(data['y']).reshape(-1,1) #y轴数据
v = np.real(data['t']).reshape(-1,1) #x轴数据
t0 = int(re.findall(r'\d+', filename)[0])
# x0 = data[:, 0].reshape(-1, 1)
# y = data[:, 1].reshape(-1, 1) # 输出 y 也需要是二维数组 (样本数, 目标数)
# x0 = x0 - min(x0) + 1.1
letter = re.sub(r'\d+', '', filename)


# p = p/sum(p)
# if letter == 'tr':
#     y = y/sum(y)
    # elif letter == 'br'
# y_smoothed = gaussian_filter1d(y, sigma=1)
# v = v/60
# X = s.reshape(-1, 1)
# y = fs.reshape(-1, 1)
# plt.semilogy(v,p,'o')
if filename == 'tr2' or filename == 'sediment01' :p = p/sum(p)
plt.loglog(v,p,'o')
# plt.figure()
# plt.plot(x0,y_smoothed,'--')
plt.xlabel('$v$')
plt.ylabel('$p_u(v)$')
plt.show()
#%%
model = PySRRegressor.from_file(    
    run_directory='outputs/20250601_162219_rHV0my' # saa b=6
    # run_directory="outputs/20250530_211324_JKcF3N" # tr b=6
    # run_directory="outputs/20250530_205411_dPkbo7" # br b=6
    # run_directory='outputs/20250819_220714_kMsUK8' # sediment power-law
      # run_directory='outputs/20250921_145323_PVKi26' # river051
    )
b = 0 # 自行选择第几个公式，=0时选择最高分公式
if b == 0:
    print('\n Sym reg:', model.sympy())
    f = abs(model.predict(v,))
else:
    print('\n Sym reg:', model.sympy(b))
    f = abs(model.predict(v,b))
# plt.plot(x0,y,'o')
# plt.plot(x0,f)
plt.figure()
plt.semilogy(v,p,'o', label='Synthetic data')
# plt.plot(v,p,'o', label='Transformed data')
# plt.loglog(v,p,'o', label='Transformed data')
plt.plot(v,f, label='Recovered expression')
# plt.plot(x0,f)
# 设置 y 轴范围（示例：限制在 [y_min, y_max]）
# y_min = 1e-3  # 最小 y 值（根据数据调整）
# y_max = np.inf   # 最大 y 值（根据数据调整）
# plt.ylim(y_min)  # 关键：限制 y 轴范围
plt.xlabel('$v$')
plt.ylabel('$p_u(v)$')
plt.legend()
plt.show()