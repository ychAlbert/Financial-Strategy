import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from utils import DL_optimize
from dl_utils import train_dl_model, DLKalmanFilter
from sklearn.linear_model import LinearRegression
import torch

import warnings
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.")

# 算法标志
ALGO_FLAG = 'DL_BCOT'  # 可选: 'DL', 'DL_KL', 'DL_OT', 'DL_BCOT', 'KL', 'OT', 'BCOT'

# 交易成本百分比
trcost_perc = 0.0001
# 初始资金
init_wealth = 10000.0

# m是观测维度
m_dim = 1
# n是未观测状态维度
n_dim = 2
# 用于估计线性方程初始系数的点数
burn_len = 100
# 用于估计价差的滚动窗口中的点数
window_size = 20

# 鲁棒性半径
radius = 1.0

# 创建模型目录
os.makedirs('./models', exist_ok=True)

def main():
    print(f'使用算法: {ALGO_FLAG}, 鲁棒性半径: {radius}')
    
    # 加载数据
    Y = pd.read_csv('AMZN.csv')
    Y_Close = Y.loc[2012-burn_len:, 'Adj Close'].values
    X = pd.read_csv('GOOG.csv')
    X_Close = X.loc[2012-burn_len:, 'Adj Close'].values
    
    # 未观测状态的转移矩阵
    A = np.eye(n_dim)
    # 观测矩阵
    C = np.zeros((m_dim, n_dim))
    
    # 初始均值和协方差
    reg = LinearRegression().fit(X_Close[:burn_len].reshape(burn_len, 1), Y_Close[:burn_len])
    init_mean = np.array([reg.intercept_, reg.coef_[0]]).reshape((n_dim, 1))
    init_cov = np.array([[1.0, 0.0], [0.0, 1.0]])
    
    # 协方差
    Bp = np.eye(2)
    Dp = np.array([1.0]).reshape((m_dim, m_dim))
    
    # 训练深度学习模型
    print('准备训练数据...')
    # 准备训练数据
    Y_train = Y_Close[:burn_len]
    X_train = X_Close[:burn_len]
    
    # 构建观测和状态序列
    observations = []
    states = []
    
    # 初始状态
    current_state = init_mean.reshape(-1)
    
    for i in range(1, burn_len):
        # 当前观测
        C[0, 0] = 1.0
        C[0, 1] = X_train[i]
        obs = np.array([Y_train[i]])
        observations.append(obs)
        
        # 下一个状态 (使用线性回归模型估计)
        next_state = np.array([reg.intercept_, reg.coef_[0]])
        states.append(next_state)
        
        # 更新当前状态
        current_state = next_state
    
    observations = np.array(observations)
    states = np.array(states)
    
    # 训练深度学习模型
    print('训练深度学习模型...')
    model_path = f'./models/dl_kalman_{ALGO_FLAG}.pt'
    
    # 检查是否已有训练好的模型
    if not os.path.exists(model_path):
        dkf = train_dl_model(
            observations, 
            states, 
            m_dim, 
            n_dim, 
            hidden_dim=64, 
            epochs=100, 
            batch_size=32, 
            save_path=model_path
        )
        print(f'模型已保存到 {model_path}')
    else:
        print(f'使用已有模型 {model_path}')
    
    # 使用深度学习模型进行滤波
    print('使用深度学习模型进行滤波...')
    
    # 准备测试数据
    Y_test = Y_Close[burn_len:]
    X_test = X_Close[burn_len:]
    horizon = len(Y_test)
    
    # 初始化
    pre_mean = init_mean.copy()
    pre_cov = init_cov.copy()
    
    # 观测
    obs = np.zeros((horizon, m_dim))
    # 滤波后的未观测状态
    est_state = np.zeros((horizon, n_dim))
    est_cov = np.zeros((horizon, n_dim, n_dim))
    
    # 滤波
    for step in range(horizon):
        obs[step, 0] = Y_test[step]
        C[0, 0] = 1.0
        C[0, 1] = X_test[step]
        
        # 使用深度学习模型进行优化
        next_mean, next_cov = DL_optimize(
            m_dim, n_dim, radius, A, Bp, C, Dp, pre_cov,
            obs[step, :].reshape((m_dim, 1)), pre_mean, 20,
            algo=ALGO_FLAG
        )
        
        est_state[step, :] = next_mean.reshape(-1)
        est_cov[step, :] = next_cov
        
        pre_mean = next_mean.copy()
        pre_cov = next_cov.copy()
        
        if step % 20 == 0:
            print(f'已滤波步骤 {step}/{horizon}')
    
    # 估计值
    estimated = est_state[:, 0] + np.multiply(est_state[:, 1], X_test)
    
    # 交易
    print('执行配对交易策略...')
    idx = window_size
    spread = Y_test - estimated
    
    open_thres = 2.0
    close_thres = 0.0
    
    position = None
    # 0表示Y，1表示X
    stock_poi = np.zeros((2, horizon))
    # 股票交易量
    quantity = 100
    cash = np.zeros(horizon)
    cash[:idx] = init_wealth
    stock_value = np.zeros(horizon)
    
    while idx < horizon:
        roll_mean = spread[idx-window_size:idx].mean()
        roll_std = spread[idx-window_size:idx].std()
        residual = spread[idx] - roll_mean
        
        if position is None:
            if residual <= open_thres*roll_std and residual >= -open_thres*roll_std:
                # 信号未触发
                cash[idx] = cash[idx-1]
            elif residual < -open_thres*roll_std:
                # 开多头仓位
                stock_poi[0, idx] = quantity
                stock_poi[1, idx] = -est_state[idx, 1] * quantity
                poi0_sign = np.sign(stock_poi[0, idx])
                poi1_sign = np.sign(stock_poi[1, idx])
                cash[idx] = cash[idx - 1] - \
                            (stock_poi[0, idx] * Y_test[idx] * (1 + poi0_sign * trcost_perc) +
                             stock_poi[1, idx] * X_test[idx] * (1 + poi1_sign * trcost_perc))
                stock_value[idx] = stock_poi[0, idx] * Y_test[idx] + stock_poi[1, idx] * X_test[idx]
                position = "long"
            elif residual > open_thres*roll_std:
                # 开空头仓位
                stock_poi[0, idx] = -quantity
                stock_poi[1, idx] = est_state[idx, 1] * quantity
                poi0_sign = np.sign(stock_poi[0, idx])
                poi1_sign = np.sign(stock_poi[1, idx])
                cash[idx] = cash[idx - 1] - \
                            (stock_poi[0, idx] * Y_test[idx] * (1 + poi0_sign * trcost_perc) +
                             stock_poi[1, idx] * X_test[idx] * (1 + poi1_sign * trcost_perc))
                stock_value[idx] = stock_poi[0, idx] * Y_test[idx] + stock_poi[1, idx] * X_test[idx]
                position = "short"
        
        elif position == "long":
            if residual < -close_thres*roll_std:
                # 维持仓位
                stock_poi[:, idx] = stock_poi[:, idx - 1]
                cash[idx] = cash[idx - 1]
                stock_value[idx] = stock_poi[0, idx] * Y_test[idx] + stock_poi[1, idx] * X_test[idx]
            else:
                # 平仓
                poi0_sign = np.sign(stock_poi[0, idx-1])
                poi1_sign = np.sign(stock_poi[1, idx-1])
                cash[idx] = cash[idx-1] + \
                            stock_poi[0, idx-1] * Y_test[idx] * (1 - poi0_sign * trcost_perc) + \
                            stock_poi[1, idx-1] * X_test[idx] * (1 - poi1_sign * trcost_perc)
                stock_poi[:, idx] = 0
                stock_value[idx] = 0
                position = None
        
        else:  # position == "short"
            if residual > close_thres*roll_std:
                # 维持仓位
                stock_poi[:, idx] = stock_poi[:, idx - 1]
                cash[idx] = cash[idx - 1]
                stock_value[idx] = stock_poi[0, idx] * Y_test[idx] + stock_poi[1, idx] * X_test[idx]
            else:
                # 平仓
                poi0_sign = np.sign(stock_poi[0, idx-1])
                poi1_sign = np.sign(stock_poi[1, idx-1])
                cash[idx] = cash[idx-1] + \
                            stock_poi[0, idx-1] * Y_test[idx] * (1 - poi0_sign * trcost_perc) + \
                            stock_poi[1, idx-1] * X_test[idx] * (1 - poi1_sign * trcost_perc)
                stock_poi[:, idx] = 0
                stock_value[idx] = 0
                position = None
        
        idx += 1
    
    # 创建日志目录
    sub_folder = f'{ALGO_FLAG}_{round(radius, 2)}'
    log_dir = f'./logs/{sub_folder}'
    os.makedirs(log_dir, exist_ok=True)
    
    # 保存参数配置
    with open(f'{log_dir}/params.txt', 'w') as fp:
        fp.write('参数设置 \n')
        fp.write(f'算法: {ALGO_FLAG} \n')
        fp.write(f'Bp: {Bp} \n')
        fp.write(f'Dp: {Dp} \n')
        fp.write(f'init_mean: {init_mean} \n')
        fp.write(f'init_cov: {init_cov} \n')
        fp.write(f'radius: {radius} \n')
        fp.write(f'horizon: {horizon} \n')
        fp.write(f'open thres: {open_thres} \n')
        fp.write(f'close thres: {close_thres} \n')
        fp.write(f'stock trading quantity: {quantity} \n')
    
    # 绘制结果
    plt.figure(1)
    plt.plot(cash + stock_value, label='total')
    plt.legend(loc='best')
    plt.savefig(f'{log_dir}/portfolio.pdf', format='pdf', dpi=500, bbox_inches='tight', pad_inches=0.1)
    
    plt.figure(2)
    plt.plot(est_state[:, 1], label='hedge ratio')
    plt.legend(loc='best')
    plt.savefig(f'{log_dir}/hedgeratio.pdf', format='pdf', dpi=500, bbox_inches='tight', pad_inches=0.1)
    
    # 保存结果
    with open(f'{log_dir}/est_state.pickle', 'wb') as fp:
        pickle.dump(est_state, fp)
    
    with open(f'{log_dir}/spread.pickle', 'wb') as fp:
        pickle.dump(spread, fp)
    
    with open(f'{log_dir}/cash.pickle', 'wb') as fp:
        pickle.dump(cash, fp)
    
    with open(f'{log_dir}/stock_value.pickle', 'wb') as fp:
        pickle.dump(stock_value, fp)
    
    with open(f'{log_dir}/position.pickle', 'wb') as fp:
        pickle.dump(stock_poi, fp)
    
    # 计算Sharpe和Sortino比率
    ptf = cash + stock_value
    rtn = np.divide(ptf[1:] - ptf[:-1], ptf[:-1])
    rtn_mean = rtn.mean()
    so_idx = rtn < rtn_mean
    sharpe_ratio = (rtn_mean - 0.02/252)/rtn.std()*np.sqrt(252)
    sortino_ratio = (rtn_mean - 0.02/252)/rtn[so_idx].std()*np.sqrt(252)
    terminal_wealth = ptf[-1]
    
    print(f'策略Sharpe比率: {sharpe_ratio:.4f}')
    print(f'策略Sortino比率: {sortino_ratio:.4f}')
    print(f'终端财富: {terminal_wealth:.2f}')
    
    # 计算基准资产的表现
    X_rtn = np.divide(X_test[1:] - X_test[:-1], X_test[:-1])
    X_m = X_rtn.mean()
    X_idx = X_rtn < X_m
    X_sharpe = (X_m - 0.02/252)/X_rtn.std()*np.sqrt(252)
    X_sortino = (X_m - 0.02/252)/X_rtn[X_idx].std()*np.sqrt(252)
    X_terminal = X_test[-1]/X_test[0]*init_wealth
    
    print(f'资产1 Sharpe比率: {X_sharpe:.4f}')
    print(f'资产1 Sortino比率: {X_sortino:.4f}')
    print(f'资产1 终端财富: {X_terminal:.2f}')
    
    Y_rtn = np.divide(Y_test[1:] - Y_test[:-1], Y_test[:-1])
    Y_m = Y_rtn.mean()
    Y_idx = Y_rtn < Y_m
    Y_sharpe = (Y_m - 0.02/252)/Y_rtn.std()*np.sqrt(252)
    Y_sortino = (Y_m - 0.02/252)/Y_rtn[Y_idx].std()*np.sqrt(252)
    Y_terminal = Y_test[-1]/Y_test[0]*init_wealth
    
    print(f'资产2 Sharpe比率: {Y_sharpe:.4f}')
    print(f'资产2 Sortino比率: {Y_sortino:.4f}')
    print(f'资产2 终端财富: {Y_terminal:.2f}')
    
    # 保存性能指标
    with open(f'{log_dir}/performance.txt', 'w') as fp:
        fp.write(f'策略Sharpe比率: {sharpe_ratio:.4f}\n')
        fp.write(f'策略Sortino比率: {sortino_ratio:.4f}\n')
        fp.write(f'终端财富: {terminal_wealth:.2f}\n')
        fp.write(f'资产1 Sharpe比率: {X_sharpe:.4f}\n')
        fp.write(f'资产1 Sortino比率: {X_sortino:.4f}\n')
        fp.write(f'资产1 终端财富: {X_terminal:.2f}\n')
        fp.write(f'资产2 Sharpe比率: {Y_sharpe:.4f}\n')
        fp.write(f'资产2 Sortino比率: {Y_sortino:.4f}\n')
        fp.write(f'资产2 终端财富: {Y_terminal:.2f}\n')
    
    return sharpe_ratio, sortino_ratio, terminal_wealth


if __name__ == "__main__":
    main()