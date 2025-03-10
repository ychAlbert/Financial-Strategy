import numpy as np
from utils import optimize
import time
import pickle
import os
from pykalman import KalmanFilter
import warnings
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.")

class bcolors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    PINK = '\033[35m'
    GREY = '\033[36m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

ALGO_FLAG = 'OT'

if ALGO_FLAG == 'BCOT':
    log_initial = 'BCOT_track_radi'
elif ALGO_FLAG == 'OT':
    log_initial = 'OT_track_radi'
elif ALGO_FLAG == 'KL':
    log_initial = 'KL_track_radi'

# Max iteration in scipy.optimize
if ALGO_FLAG == 'KL':
    MaxIter = 2       # KL in this implementation is easy to diverge, MaxIter = 2 is the best choice
else:
    MaxIter = 20
start = time.time()
np.random.seed(12345)
# pre-training data length
pre_sample_size = 10
# Simulation runs for a given radius
n_ins = 10
# Step length of real data (T) in the paper
step_num = 100

search_num = 8

#radi_arr = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
radi_arr = np.array([4.0])
# obs dimension
m_dim = 2
# state dimension
n_dim = 4
dt = 1

A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
C = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
r = 50
q = 10
def simulator(step_num):
    T = step_num
    unobs_state = np.zeros((step_num, n_dim))
    obs = np.zeros((step_num, m_dim))

    D_true = (0.1 + 0.05*np.cos(np.pi*0/T))*r*np.array([[1, 0.5], [0.5, 1]])

    unobs_state[0, :] = np.random.multivariate_normal(mean=np.zeros(n_dim), cov=np.eye(n_dim), size=1)
    obs[0, :] = C @ unobs_state[0, :] + np.random.multivariate_normal(np.zeros(m_dim), D_true, size=1)

    for step in range(1, step_num):
        B_temp = np.array([[dt ** 3 / 3, 0, dt ** 2 / 2, 0], [0, dt ** 3 / 3, 0, dt ** 2 / 2],
                           [dt ** 2 / 2, 0, dt, 0], [0, dt ** 2 / 2, 0, dt]])*q
        B_true = (6.5 + 0.5*np.cos(np.pi*step/T))*B_temp
        unobs_state[step, :] = A @ unobs_state[step-1, :] + \
                               np.random.multivariate_normal(mean=np.zeros(n_dim), cov=B_true, size=1)

        D_true = (0.1 + 0.05*np.cos(np.pi*step/T))*r*np.array([[1, 0.5], [0.5, 1]])
        obs[step, :] = C @ unobs_state[step, :] + \
                       np.random.multivariate_normal(np.zeros(m_dim), D_true, size=1)
    return unobs_state, obs




for r_idx in range(search_num):
    radius = radi_arr[r_idx]
    print('Testing radius', r_idx, radius)

    poi_err = np.zeros((n_ins, step_num))
    vel_err = np.zeros((n_ins, step_num))

    EM_poi_err = np.zeros((n_ins, step_num))
    EM_vel_err = np.zeros((n_ins, step_num))

    obs_hist = []
    unobs_hist = []
    EM_est_hist = []
    algo_est_hist = []

    for ins in np.arange(n_ins):
        print('Instance', ins)
        # dB = np.zeros(step_num)
        # Simulate a sample used for kf estimation of noise covariance
        pretrain_unobs_state, pretrain_obs = simulator(pre_sample_size)
        kf = KalmanFilter(transition_matrices=A, observation_matrices=C)
        kf = kf.em(pretrain_obs)
        B_nom = kf.transition_covariance
        D_nom = kf.observation_covariance
        pre_mean = kf.initial_state_mean.reshape((n_dim, 1))
        pre_cov = kf.initial_state_covariance

        (EM_pre_est, filtered_state_covariances) = kf.filter(pretrain_obs)

        #####################################################
        MSE = np.sum((EM_pre_est[:, :2] - pretrain_unobs_state[:, :2]) ** 2, axis=1).mean()
        print('EM样本内位置误差RMSE为', np.sqrt(MSE))

        MSE = np.sum((EM_pre_est[:, 2:] - pretrain_unobs_state[:, 2:]) ** 2, axis=1).mean()
        print('EM样本内速度误差RMSE为', np.sqrt(MSE))

        ########### 样本外测试 ###############
        unobs_state, obs = simulator(step_num)

        kf_out = KalmanFilter(transition_matrices=A, observation_matrices=C,
                              transition_covariance=B_nom, observation_covariance=D_nom,
                              initial_state_mean=kf.initial_state_mean, initial_state_covariance=pre_cov)
        # kf_out = kf_out.em(obs)

        (EM_est, filtered_state_covariances) = kf_out.filter(obs)

        #####################################################
        EM_poi_err[ins, :] = np.sum((EM_est[:, :2] - unobs_state[:, :2]) ** 2, axis=1)
        print(bcolors.PINK + 'EM样本外位置误差RMSE为',
              np.sqrt(EM_poi_err[ins, :].mean()), bcolors.ENDC)

        EM_vel_err[ins, :] = np.sum((EM_est[:, 2:] - unobs_state[:, 2:]) ** 2, axis=1)
        print(bcolors.PINK + 'EM样本外速度误差RMSE为',
              np.sqrt(EM_vel_err[ins, :].mean()), bcolors.ENDC)

        est_state = np.zeros((step_num, n_dim))
        est_cov = np.zeros((step_num, n_dim, n_dim))

        for step in range(step_num):
            next_mean, next_cov = optimize(m_dim, n_dim, radius, A, B_nom, C, D_nom, pre_cov,
                                           obs[step, :].reshape((m_dim, 1)), pre_mean, MaxIter,
                                           algo=ALGO_FLAG)

            est_state[step, :] = next_mean.reshape(-1)
            est_cov[step, :] = next_cov

            pre_mean = next_mean.copy()
            pre_cov = next_cov.copy()

            # dB[step] = 10*np.log10(np.sum((unobs_state[step, :] - est_state[step, :])**2))
            poi_err[ins, step] = np.sum((unobs_state[step, :2] - est_state[step, :2])**2)
            vel_err[ins, step] = np.sum((unobs_state[step, 2:] - est_state[step, 2:])**2)

        MSE = np.sum((est_state[:, :2] - unobs_state[:, :2])**2, axis=1).mean()
        print(bcolors.GREEN + '算法位置误差为', np.sqrt(MSE), bcolors.ENDC)

        MSE = np.sum((est_state[:, 2:] - unobs_state[:, 2:])**2, axis=1).mean()
        print(bcolors.GREEN + '算法速度误差为', np.sqrt(MSE), bcolors.ENDC)

        obs_hist.append(obs.copy())
        unobs_hist.append(unobs_state.copy())
        EM_est_hist.append(EM_est.copy())
        algo_est_hist.append(est_state.copy())

        end = time.time()
        print('耗时', end - start)


    log_dir = "./logs/{}_{}".format(log_initial, radius)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Save params configuration
    with open('{}/params.txt'.format(log_dir), 'w') as fp:
        fp.write('Params setting \n')
        fp.write('radius {}, n_ins {}, step_num {} \n'.format(radius, n_ins, step_num))
        fp.write('Maxiter = {} \n'.format(MaxIter))

    with open('{}/obs.pickle'.format(log_dir), 'wb') as fp:
        pickle.dump(obs_hist, fp)

    with open('{}/est_state.pickle'.format(log_dir), 'wb') as fp:
        pickle.dump(algo_est_hist, fp)

    with open('{}/unobs_state.pickle'.format(log_dir), 'wb') as fp:
        pickle.dump(unobs_hist, fp)

    with open('{}/EM_est.pickle'.format(log_dir), 'wb') as fp:
        pickle.dump(EM_est_hist, fp)