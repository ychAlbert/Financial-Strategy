import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from utils import optimize
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.")

# Transaction cost percentage
trcost_perc = 0.0001
# Initial wealth
init_wealth = 10000.0

# m is dim of observations
m_dim = 1
# n is dim of unobserved states
n_dim = 2
# points used to estimate initial coef in linear equation
burn_len = 100
# points used in rolling window of spread estimation
window_size = 20

ALGO_FLAG = 'BCOT'

if ALGO_FLAG == 'Nonrobust':
    search_num = 1
else:
    search_num = 10

if ALGO_FLAG == 'KL':
    MaxIter = 2     #
else:
    MaxIter = 20

# radi_arr = np.linspace(0.01, 0.2, search_num)

radi_arr = np.linspace(0.1, 1.0, search_num)
sharpe = np.zeros_like(radi_arr)
sortino = np.zeros_like(radi_arr)
terminal_wealth = np.zeros_like(radi_arr)

for r_idx in range(search_num):

    radius = radi_arr[r_idx]
    print('Testing radius', radius)

    Y = pd.read_csv('AMZN.csv')
    Y_Close = Y.loc[2012-burn_len:, 'Adj Close'].values
    X = pd.read_csv('GOOG.csv')
    X_Close = X.loc[2012-burn_len:, 'Adj Close'].values


    # transition matrix of unobserved states
    A = np.eye(n_dim)
    # transition matrix of observations
    C = np.zeros((m_dim, n_dim))


    # initial mean and cov of unobserved states
    reg = LinearRegression().fit(X_Close[:burn_len].reshape(burn_len, 1), Y_Close[:burn_len])
    init_mean = np.array([reg.intercept_, reg.coef_[0]]).reshape((n_dim, 1))
    init_cov = np.array([[1.0, 0.0], [0.0, 1.0]])

    # covs
    Bp = np.eye(2)
    Dp = np.array([1.0]).reshape((m_dim, m_dim))


    pre_mean = init_mean.copy()
    pre_cov = init_cov.copy()

    # total time steps, including burn-in periods
    Y_Close = Y_Close[burn_len:]
    X_Close = X_Close[burn_len:]
    horizon = len(Y_Close)
    # observations
    obs = np.zeros((horizon, m_dim))
    # filtered unobserved states
    est_state = np.zeros((horizon, n_dim))
    est_cov = np.zeros((horizon, n_dim, n_dim))

    ################### filtering ######################
    for step in range(horizon):
        obs[step, 0] = Y_Close[step]
        C[0, 0] = 1.0
        C[0, 1] = X_Close[step]
        next_mean, next_cov = optimize(m_dim, n_dim, radius, A, Bp, C, Dp, pre_cov,
                                       obs[step, :].reshape((m_dim, 1)), pre_mean, MaxIter,
                                       algo=ALGO_FLAG)

        est_state[step, :] = next_mean.reshape(-1)
        est_cov[step, :] = next_cov

        pre_mean = next_mean.copy()
        pre_cov = next_cov.copy()
        # if step%5 == 0:
        #     print('Filtered step', step)
        #     print('Estimated det', np.linalg.det(next_cov))
            # if np.linalg.det(next_cov) < 0.0:
            #     print(res.constr_violation,' CG stop cond:', res.cg_stop_cond, 'Status:', res.status)

    estimated = est_state[:, 0] + np.multiply(est_state[:, 1], X_Close)


    ########### trading ##############

    idx = window_size
    spread = Y_Close - estimated

    open_thres = 2.0
    close_thres = 0.0

    position = None
    # 0 for Y, 1 for X
    stock_poi = np.zeros((2, horizon))
    # stock trading volume
    quantity = 100
    cash = np.zeros(horizon)
    cash[:idx] = init_wealth
    stock_value = np.zeros(horizon)

    while idx < horizon:
        roll_mean = spread[idx-window_size:idx].mean()
        roll_std = spread[idx-window_size:idx].std()
        residual = spread[idx] - roll_mean

        if position == None:
            if residual <= open_thres*roll_std and residual >= -open_thres*roll_std:
                # Signal not triggered.
                cash[idx] = cash[idx-1]
            elif residual < -open_thres*roll_std:
                # open long position
                stock_poi[0, idx] = quantity
                stock_poi[1, idx] = -est_state[idx, 1] * quantity
                poi0_sign = np.sign(stock_poi[0, idx])
                poi1_sign = np.sign(stock_poi[1, idx])
                cash[idx] = cash[idx - 1] - \
                            (stock_poi[0, idx] * Y_Close[idx] * (1 + poi0_sign * trcost_perc) +
                             stock_poi[1, idx] * X_Close[idx] * (1 + poi1_sign * trcost_perc))
                stock_value[idx] = stock_poi[0, idx] * Y_Close[idx] + stock_poi[1, idx] * X_Close[idx]
                position = "long"
            elif residual > open_thres*roll_std:
                # open short position
                stock_poi[0, idx] = -quantity
                stock_poi[1, idx] = est_state[idx, 1] * quantity
                poi0_sign = np.sign(stock_poi[0, idx])
                poi1_sign = np.sign(stock_poi[1, idx])
                cash[idx] = cash[idx - 1] - \
                            (stock_poi[0, idx] * Y_Close[idx] * (1 + poi0_sign * trcost_perc) +
                             stock_poi[1, idx] * X_Close[idx] * (1 + poi1_sign * trcost_perc))
                stock_value[idx] = stock_poi[0, idx] * Y_Close[idx] + stock_poi[1, idx] * X_Close[idx]
                position = "short"

        elif position == "long":
            if residual < -close_thres*roll_std:
                # maintain position
                stock_poi[:, idx] = stock_poi[:, idx - 1]
                cash[idx] = cash[idx - 1]
                stock_value[idx] = stock_poi[0, idx] * Y_Close[idx] + stock_poi[1, idx] * X_Close[idx]
            else:
                # close position
                poi0_sign = np.sign(stock_poi[0, idx-1])
                poi1_sign = np.sign(stock_poi[1, idx-1])
                cash[idx] = cash[idx-1] + \
                            stock_poi[0, idx-1] * Y_Close[idx] * (1 - poi0_sign * trcost_perc) + \
                            stock_poi[1, idx-1] * X_Close[idx] * (1 - poi1_sign * trcost_perc)
                stock_poi[:, idx] = 0
                stock_value[idx] = 0
                position = None

        else:
            if residual > close_thres*roll_std:
                # maintain position
                stock_poi[:, idx] = stock_poi[:, idx - 1]
                cash[idx] = cash[idx - 1]
                stock_value[idx] = stock_poi[0, idx] * Y_Close[idx] + stock_poi[1, idx] * X_Close[idx]
            else:
                # close position
                poi0_sign = np.sign(stock_poi[0, idx-1])
                poi1_sign = np.sign(stock_poi[1, idx-1])
                cash[idx] = cash[idx-1] + \
                            stock_poi[0, idx-1] * Y_Close[idx] * (1 - poi0_sign * trcost_perc) + \
                            stock_poi[1, idx-1] * X_Close[idx] * (1 - poi1_sign * trcost_perc)
                stock_poi[:, idx] = 0
                stock_value[idx] = 0
                position = None

        idx += 1



    if ALGO_FLAG != 'Nonrobust':
        sub_folder = '{}_{}'.format(ALGO_FLAG, round(radius, 2))

        log_dir = './logs/{}'.format(sub_folder)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Save params configuration
        with open('{}/params.txt'.format(log_dir), 'w') as fp:
            fp.write('Params setting \n')
            fp.write('Algorithm: {} \n'.format(ALGO_FLAG))
            fp.write('Bp: {} \n'.format(Bp))
            fp.write('Dp: {} \n'.format(Dp))
            fp.write('init_mean: {} \n'.format(init_mean))
            fp.write('init_cov: {} \n'.format(init_cov))
            fp.write('radius: {} \n'.format(radius))
            fp.write('horizon: {} \n'.format(horizon))
            fp.write('open thres: {} \n'.format(open_thres))
            fp.write('close thres: {} \n'.format(close_thres))
            fp.write('stock trading quantity: {} \n'.format(quantity))
            fp.write('maxiter: {} \n'.format(MaxIter))

        plt.figure(1)
        plt.plot(cash + stock_value, label='total')
        # plt.plot(cash, label='cash')
        # plt.plot(stock_value, label='stock')
        plt.legend(loc='best')
        plt.savefig('{}/portfolio.pdf'.format(log_dir), format='pdf', dpi=500, bbox_inches='tight', pad_inches=0.1)

        plt.figure(2)
        plt.plot(est_state[:, 1], label='hedge ratio')
        plt.legend(loc='best')
        plt.savefig('{}/hedgeratio.pdf'.format(log_dir), format='pdf', dpi=500, bbox_inches='tight', pad_inches=0.1)

        with open('{}/est_state.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(est_state, fp)

        with open('{}/spread.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(spread, fp)

        with open('{}/cash.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(cash, fp)

        with open('{}/stock_value.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(stock_value, fp)

        with open('{}/position.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(stock_poi, fp)

    ##### Calculate Sharpe and Sortino ratios
    ptf = cash + stock_value
    rtn = np.divide(ptf[1:] - ptf[:-1], ptf[:-1])
    rtn_mean = rtn.mean()
    so_idx = rtn < rtn_mean
    sharpe[r_idx] = (rtn_mean - 0.02/252)/rtn.std()*np.sqrt(252)
    sortino[r_idx] = (rtn_mean - 0.02/252)/rtn[so_idx].std()*np.sqrt(252)

    terminal_wealth[r_idx] = ptf[-1]

    print('Sharpe ratio of strategy:', sharpe[r_idx])
    print('Sortino ratio of strategy:', sortino[r_idx])
    print('Terminal wealth:', terminal_wealth[r_idx])

    X_rtn = np.divide(X_Close[1:] - X_Close[:-1], X_Close[:-1])
    X_m = X_rtn.mean()
    X_idx = X_rtn < X_m
    X_sharpe = (X_m - 0.02/252)/X_rtn.std()*np.sqrt(252)
    X_sortino = (X_m - 0.02/252)/X_rtn[X_idx].std()*np.sqrt(252)
    X_terminal = X_Close[-1]/X_Close[0]*init_wealth
    print('Asset 1 Sharpe ratio:', X_sharpe)
    print('Asset 1 Sortino ratio:', X_sortino)
    print('Terminal wealth if invested in Asset 1 only', X_terminal)

    Y_rtn = np.divide(Y_Close[1:] - Y_Close[:-1], Y_Close[:-1])
    Y_m = Y_rtn.mean()
    Y_idx = Y_rtn < Y_m
    Y_sharpe = (Y_m - 0.02/252)/Y_rtn.std()*np.sqrt(252)
    Y_sortino = (Y_m - 0.02/252)/Y_rtn[Y_idx].std()*np.sqrt(252)
    Y_terminal = Y_Close[-1]/Y_Close[0]*init_wealth
    print('Asset 2 Sharpe ratio:', Y_sharpe)
    print('Asset 2 Sortino ratio:', Y_sortino)
    print('Terminal wealth if invested in Asset 2 only', Y_terminal)

print('Sharpe')
print(np.round(sharpe, 4))
print('Sortino')
print(np.round(sortino, 4))
print('Terminal wealth')
print(np.round(terminal_wealth, 0))

if ALGO_FLAG != 'Nonrobust':
    with open('./logs/sharpe_{}.pickle'.format(ALGO_FLAG), 'wb') as fp:
        pickle.dump(sharpe, fp)

    with open('./logs/sortino_{}.pickle'.format(ALGO_FLAG), 'wb') as fp:
        pickle.dump(sortino, fp)


def main():
    """主函数，用于比较脚本调用
    
    Returns:
        sharpe_ratio: Sharpe比率
        sortino_ratio: Sortino比率
        terminal_wealth: 终端财富
    """
    global ALGO_FLAG, radius
    
    # 使用单一半径运行
    r_idx = 0
    if isinstance(radius, float):
        # 如果radius已经是单一值
        radi_arr = np.array([radius])
        sharpe = np.zeros(1)
        sortino = np.zeros(1)
        terminal_wealth = np.zeros(1)
    else:
        # 使用第一个半径值
        radius = radi_arr[0]
    
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

    pre_mean = init_mean.copy()
    pre_cov = init_cov.copy()

    # 总时间步长，包括预热期
    Y_Close = Y_Close[burn_len:]
    X_Close = X_Close[burn_len:]
    horizon = len(Y_Close)
    # 观测
    obs = np.zeros((horizon, m_dim))
    # 滤波后的未观测状态
    est_state = np.zeros((horizon, n_dim))
    est_cov = np.zeros((horizon, n_dim, n_dim))

    # 滤波
    for step in range(horizon):
        obs[step, 0] = Y_Close[step]
        C[0, 0] = 1.0
        C[0, 1] = X_Close[step]
        next_mean, next_cov = optimize(m_dim, n_dim, radius, A, Bp, C, Dp, pre_cov,
                                    obs[step, :].reshape((m_dim, 1)), pre_mean, MaxIter,
                                    algo=ALGO_FLAG)

        est_state[step, :] = next_mean.reshape(-1)
        est_cov[step, :] = next_cov

        pre_mean = next_mean.copy()
        pre_cov = next_cov.copy()

    # 估计值
    estimated = est_state[:, 0] + np.multiply(est_state[:, 1], X_Close)

    # 交易
    idx = window_size
    spread = Y_Close - estimated

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

        if position == None:
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
                            (stock_poi[0, idx] * Y_Close[idx] * (1 + poi0_sign * trcost_perc) +
                            stock_poi[1, idx] * X_Close[idx] * (1 + poi1_sign * trcost_perc))
                stock_value[idx] = stock_poi[0, idx] * Y_Close[idx] + stock_poi[1, idx] * X_Close[idx]
                position = "long"
            elif residual > open_thres*roll_std:
                # 开空头仓位
                stock_poi[0, idx] = -quantity
                stock_poi[1, idx] = est_state[idx, 1] * quantity
                poi0_sign = np.sign(stock_poi[0, idx])
                poi1_sign = np.sign(stock_poi[1, idx])
                cash[idx] = cash[idx - 1] - \
                            (stock_poi[0, idx] * Y_Close[idx] * (1 + poi0_sign * trcost_perc) +
                            stock_poi[1, idx] * X_Close[idx] * (1 + poi1_sign * trcost_perc))
                stock_value[idx] = stock_poi[0, idx] * Y_Close[idx] + stock_poi[1, idx] * X_Close[idx]
                position = "short"

        elif position == "long":
            if residual < -close_thres*roll_std:
                # 维持仓位
                stock_poi[:, idx] = stock_poi[:, idx - 1]
                cash[idx] = cash[idx - 1]
                stock_value[idx] = stock_poi[0, idx] * Y_Close[idx] + stock_poi[1, idx] * X_Close[idx]
            else:
                # 平仓
                poi0_sign = np.sign(stock_poi[0, idx-1])
                poi1_sign = np.sign(stock_poi[1, idx-1])
                cash[idx] = cash[idx-1] + \
                            stock_poi[0, idx-1] * Y_Close[idx] * (1 - poi0_sign * trcost_perc) + \
                            stock_poi[1, idx-1] * X_Close[idx] * (1 - poi1_sign * trcost_perc)
                stock_poi[:, idx] = 0
                stock_value[idx] = 0
                position = None

        else:  # position == "short"
            if residual > close_thres*roll_std:
                # 维持仓位
                stock_poi[:, idx] = stock_poi[:, idx - 1]
                cash[idx] = cash[idx - 1]
                stock_value[idx] = stock_poi[0, idx] * Y_Close[idx] + stock_poi[1, idx] * X_Close[idx]
            else:
                # 平仓
                poi0_sign = np.sign(stock_poi[0, idx-1])
                poi1_sign = np.sign(stock_poi[1, idx-1])
                cash[idx] = cash[idx-1] + \
                            stock_poi[0, idx-1] * Y_Close[idx] * (1 - poi0_sign * trcost_perc) + \
                            stock_poi[1, idx-1] * X_Close[idx] * (1 - poi1_sign * trcost_perc)
                stock_poi[:, idx] = 0
                stock_value[idx] = 0
                position = None

        idx += 1

    # 创建日志目录
    sub_folder = '{}_{}'.format(ALGO_FLAG, round(radius, 2))
    log_dir = './logs/{}'.format(sub_folder)
    os.makedirs(log_dir, exist_ok=True)

    # 保存参数配置
    with open('{}/params.txt'.format(log_dir), 'w') as fp:
        fp.write('参数设置 \n')
        fp.write('算法: {} \n'.format(ALGO_FLAG))
        fp.write('Bp: {} \n'.format(Bp))
        fp.write('Dp: {} \n'.format(Dp))
        fp.write('init_mean: {} \n'.format(init_mean))
        fp.write('init_cov: {} \n'.format(init_cov))
        fp.write('radius: {} \n'.format(radius))
        fp.write('horizon: {} \n'.format(horizon))
        fp.write('open thres: {} \n'.format(open_thres))
        fp.write('close thres: {} \n'.format(close_thres))
        fp.write('stock trading quantity: {} \n'.format(quantity))

    # 绘制结果
    plt.figure(1)
    plt.plot(cash + stock_value, label='total')
    plt.legend(loc='best')
    plt.savefig('{}/portfolio.pdf'.format(log_dir), format='pdf', dpi=500, bbox_inches='tight', pad_inches=0.1)

    plt.figure(2)
    plt.plot(est_state[:, 1], label='hedge ratio')
    plt.legend(loc='best')
    plt.savefig('{}/hedgeratio.pdf'.format(log_dir), format='pdf', dpi=500, bbox_inches='tight', pad_inches=0.1)

    # 保存结果
    with open('{}/est_state.pickle'.format(log_dir), 'wb') as fp:
        pickle.dump(est_state, fp)

    with open('{}/spread.pickle'.format(log_dir), 'wb') as fp:
        pickle.dump(spread, fp)

    with open('{}/cash.pickle'.format(log_dir), 'wb') as fp:
        pickle.dump(cash, fp)

    with open('{}/stock_value.pickle'.format(log_dir), 'wb') as fp:
        pickle.dump(stock_value, fp)

    with open('{}/position.pickle'.format(log_dir), 'wb') as fp:
        pickle.dump(stock_poi, fp)

    # 计算Sharpe和Sortino比率
    ptf = cash + stock_value
    rtn = np.divide(ptf[1:] - ptf[:-1], ptf[:-1])
    rtn_mean = rtn.mean()
    so_idx = rtn < rtn_mean
    sharpe_ratio = (rtn_mean - 0.02/252)/rtn.std()*np.sqrt(252)
    sortino_ratio = (rtn_mean - 0.02/252)/rtn[so_idx].std()*np.sqrt(252)
    terminal_wealth_value = ptf[-1]

    # 打印结果
    print('策略Sharpe比率:', sharpe_ratio)
    print('策略Sortino比率:', sortino_ratio)
    print('终端财富:', terminal_wealth_value)

    # 计算基准资产的表现
    X_rtn = np.divide(X_Close[1:] - X_Close[:-1], X_Close[:-1])
    X_m = X_rtn.mean()
    X_idx = X_rtn < X_m
    X_sharpe = (X_m - 0.02/252)/X_rtn.std()*np.sqrt(252)
    X_sortino = (X_m - 0.02/252)/X_rtn[X_idx].std()*np.sqrt(252)
    X_terminal = X_Close[-1]/X_Close[0]*init_wealth

    print('资产1 Sharpe比率:', X_sharpe)
    print('资产1 Sortino比率:', X_sortino)
    print('资产1 终端财富:', X_terminal)

    Y_rtn = np.divide(Y_Close[1:] - Y_Close[:-1], Y_Close[:-1])
    Y_m = Y_rtn.mean()
    Y_idx = Y_rtn < Y_m
    Y_sharpe = (Y_m - 0.02/252)/Y_rtn.std()*np.sqrt(252)
    Y_sortino = (Y_m - 0.02/252)/Y_rtn[Y_idx].std()*np.sqrt(252)
    Y_terminal = Y_Close[-1]/Y_Close[0]*init_wealth

    print('资产2 Sharpe比率:', Y_sharpe)
    print('资产2 Sortino比率:', Y_sortino)
    print('资产2 终端财富:', Y_terminal)
    
    # 保存性能指标
    with open(f'{log_dir}/performance.txt', 'w') as fp:
        fp.write(f'策略Sharpe比率: {sharpe_ratio:.4f}\n')
        fp.write(f'策略Sortino比率: {sortino_ratio:.4f}\n')
        fp.write(f'终端财富: {terminal_wealth_value:.2f}\n')
        fp.write(f'资产1 Sharpe比率: {X_sharpe:.4f}\n')
        fp.write(f'资产1 Sortino比率: {X_sortino:.4f}\n')
        fp.write(f'资产1 终端财富: {X_terminal:.2f}\n')
        fp.write(f'资产2 Sharpe比率: {Y_sharpe:.4f}\n')
        fp.write(f'资产2 Sortino比率: {Y_sortino:.4f}\n')
        fp.write(f'资产2 终端财富: {Y_terminal:.2f}\n')
    
    return sharpe_ratio, sortino_ratio, terminal_wealth_value


if __name__ == "__main__":
    main()

