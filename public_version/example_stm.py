# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 10:55:14 2024

@author: Xiangnan Yu

This example will show the detailed discovery process of 
memory function of Application case 4
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
# acquire the current path
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
# change the workspace to the current path
os.chdir(current_dir)
# import numpy as np

filename = 'br2' 
'''
filename: br2, tr2 and sediment01 are used to symbolic regression
'''
data = io.loadmat(f'dataset/{filename}.mat')
'The data has been '
p = np.real(data['y']).reshape(-1,1) # pdf
v = np.real(data['t']).reshape(-1,1) # velocity magnitude
t0 = int(re.findall(r'\d+', filename)[0])
# x0 = data[:, 0].reshape(-1, 1)
# y = data[:, 1].reshape(-1, 1) 
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
p = p/sum(p)
plt.loglog(v,p,'o')
# plt.figure()
# plt.plot(x0,y_smoothed,'--')
plt.xlabel('$v$')
plt.ylabel('$p_u(v)$')
plt.show()
#%%
# weighted loss
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
# log loss
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
#%
model = PySRRegressor(
    # run_directory = 'outputs\20250530_211324_JKcF3N',
    model_selection="best",  # Result is mix of simplicity+accuracy
    maxsize = 15,
    # denoise=True,
    niterations = 2000,
    binary_operators=["*", "/", "^"
                      , "+"
                      ],
    unary_operators=[
    #     # "cos",
        "exp",
        # "log",
        # "inv(x) = 1/x",
    #     # ^ Custom operator (julia syntax)
        # "stable_pdf",
    ],
    # extra_sympy_mappings={
    #     "stable_pdf": lambda x, a, b: f"StablePDF({x}, {a}, {b})"
    # },
    # constraints={
    #     # "exp": 5,
    #     # "log": 9,
    #     "^": (-4, 4)
    #     },
    nested_constraints={
        "exp": {"exp": 0, "^": 0}, 
        # "exp": {"^": 1}, #{"sin": {"cos": 0}}, "cos": {"cos": 2}}
        "^": {"^": 0, "exp": 0},
    },
    # complexity_of_constants = 1,
    # complexity_of_operators={"exp": 0, 
    #                          # "*": 2,
    #                          },
    # extra_sympy_mappings={"inv": lambda x: 1 / x},
    # ^ Define operator for SymPy as well
    # elementwise_loss="loss(x, y) = (x - y)^2",
    # elementwise_loss="loss(x, y) = (x - y)^2/(abs(y) + 1e-3)",
    # elementwise_loss="loss(x, y) = (x - y)^2/(abs(y*(max(y)-y))+1e-3)",
    loss_function=seg_loss2,
    # ^ Custom loss function (julia syntax)
)

model.fit(v, p)
print(model,'\n')
print( '\n\n Sym reg:', model.sympy())
# %%
'''the best model in our manuscript has been stored as below:'''
# model = PySRRegressor.from_file(
#     #run_directory="outputs/20250530_211324_JKcF3N" # tr b=6
#     # run_directory="outputs/20250530_205411_dPkbo7" # br b=6
#     # run_directory='outputs/20250601_162219_rHV0my' # synthetic b=6
#     # run_directory='outputs/20250819_220714_kMsUK8' # sediment power-law

#     )
b = 0 # expression index in pareto front，best score when b=0
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
# set the scope of y-axis（[y_min, y_max]）
# y_min = 1e-3  
# y_max = np.inf  
# plt.ylim(y_min)
plt.xlabel('$v$')
plt.ylabel('$p_u(v)$')
plt.legend()
plt.show()


