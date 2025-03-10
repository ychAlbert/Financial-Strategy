import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
import os
import time
import subprocess
import warnings
from matplotlib import font_manager
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.")
# 设置支持中文的字体
font_path = 'C:/Windows/Fonts/msyh.ttc'  # 你可以根据你的系统调整字体路径
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

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

def run_tracking(algo):
    """运行tracking.py生成模拟数据并应用卡尔曼滤波器"""
    print(f"\n{bcolors.BLUE}正在运行 {algo} 算法的跟踪模拟...{bcolors.ENDC}")
    
    # 修改tracking.py中的ALGO_FLAG
    with open('tracking.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换算法标志
    content = content.replace("ALGO_FLAG = 'BCOT'", f"ALGO_FLAG = '{algo}'")
    content = content.replace("ALGO_FLAG = 'OT'", f"ALGO_FLAG = '{algo}'")
    content = content.replace("ALGO_FLAG = 'KL'", f"ALGO_FLAG = '{algo}'")
    
    # 写回文件
    with open('tracking.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    # 运行tracking.py
    start_time = time.time()
    subprocess.run(['python', 'tracking.py'], check=True)
    end_time = time.time()
    
    print(f"{bcolors.GREEN}{algo} 算法跟踪模拟完成，耗时 {end_time - start_time:.2f} 秒{bcolors.ENDC}")
def compare_position_velocity_errors():
    """比较不同算法的位置和速度误差"""
    print(f"\n{bcolors.BLUE}正在比较不同算法的位置和速度误差...{bcolors.ENDC}")
    
    # 加载不同算法的位置和速度误差数据
    radi_arr = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    n_radi = 8
    n_ins = 10
    n_step = 100
    
    # 创建存储误差数据的数组
    cot_poi_diff_all = np.zeros((n_radi, n_ins, n_step))
    ot_poi_diff_all = np.zeros((n_radi, n_ins, n_step))
    kl_poi_diff_all = np.zeros((n_radi, n_ins, n_step))
    
    cot_vel_diff_all = np.zeros((n_radi, n_ins, n_step))
    ot_vel_diff_all = np.zeros((n_radi, n_ins, n_step))
    kl_vel_diff_all = np.zeros((n_radi, n_ins, n_step))
    
    # 从各个算法的日志目录中加载数据
    for r_idx in range(n_radi):
        radius = radi_arr[r_idx]
        
        # 加载BCOT数据
        cot_dir = f"./logs/BCOT_track_radi_{radius}"
        try:
            with open(f'{cot_dir}/unobs_state.pickle', 'rb') as fp:
                unobs = pickle.load(fp)
            with open(f'{cot_dir}/est_state.pickle', 'rb') as fp:
                cot_est = pickle.load(fp)
            with open(f'{cot_dir}/EM_est.pickle', 'rb') as fp:
                EM_est = pickle.load(fp)
                
            unobs = np.array(unobs)
            cot_est = np.array(cot_est)
            EM_est = np.array(EM_est)
            
            cot_poi_err = np.sqrt(np.sum((cot_est[:, :, :2] - unobs[:, :, :2]) ** 2, axis=2))
            em_poi_err = np.sqrt(np.sum((EM_est[:, :, :2] - unobs[:, :, :2]) ** 2, axis=2))
            cot_poi_diff_all[r_idx] = cot_poi_err - em_poi_err
            
            cot_vel_err = np.sqrt(np.sum((cot_est[:, :, 2:] - unobs[:, :, 2:]) ** 2, axis=2))
            em_vel_err = np.sqrt(np.sum((EM_est[:, :, 2:] - unobs[:, :, 2:]) ** 2, axis=2))
            cot_vel_diff_all[r_idx] = cot_vel_err - em_vel_err
        except FileNotFoundError:
            print(f"警告：找不到BCOT算法在半径{radius}下的数据")
        
        # 加载OT数据
        ot_dir = f"./logs/OT_track_radi_{radius}"
        try:
            with open(f'{ot_dir}/unobs_state.pickle', 'rb') as fp:
                ot_unobs = pickle.load(fp)
            with open(f'{ot_dir}/est_state.pickle', 'rb') as fp:
                ot_est = pickle.load(fp)
            with open(f'{ot_dir}/EM_est.pickle', 'rb') as fp:
                ot_EM_est = pickle.load(fp)
                
            ot_unobs = np.array(ot_unobs)
            ot_est = np.array(ot_est)
            ot_EM_est = np.array(ot_EM_est)
            
            ot_poi_err = np.sqrt(np.sum((ot_est[:, :, :2] - ot_unobs[:, :, :2]) ** 2, axis=2))
            ot_em_poi_err = np.sqrt(np.sum((ot_EM_est[:, :, :2] - ot_unobs[:, :, :2]) ** 2, axis=2))
            ot_poi_diff_all[r_idx] = ot_poi_err - ot_em_poi_err
            
            ot_vel_err = np.sqrt(np.sum((ot_est[:, :, 2:] - ot_unobs[:, :, 2:]) ** 2, axis=2))
            ot_em_vel_err = np.sqrt(np.sum((ot_EM_est[:, :, 2:] - ot_unobs[:, :, 2:]) ** 2, axis=2))
            ot_vel_diff_all[r_idx] = ot_vel_err - ot_em_vel_err
        except FileNotFoundError:
            print(f"警告：找不到OT算法在半径{radius}下的数据")
        
        # 加载KL数据
        kl_dir = f"./logs/KL_track_radi_{radius}"
        try:
            with open(f'{kl_dir}/unobs_state.pickle', 'rb') as fp:
                kl_unobs = pickle.load(fp)
            with open(f'{kl_dir}/est_state.pickle', 'rb') as fp:
                kl_est = pickle.load(fp)
            with open(f'{kl_dir}/EM_est.pickle', 'rb') as fp:
                kl_EM_est = pickle.load(fp)
                
            kl_unobs = np.array(kl_unobs)
            kl_est = np.array(kl_est)
            kl_EM_est = np.array(kl_EM_est)
            
            kl_poi_err = np.sqrt(np.sum((kl_est[:, :, :2] - kl_unobs[:, :, :2]) ** 2, axis=2))
            kl_em_poi_err = np.sqrt(np.sum((kl_EM_est[:, :, :2] - kl_unobs[:, :, :2]) ** 2, axis=2))
            kl_poi_diff_all[r_idx] = kl_poi_err - kl_em_poi_err
            
            kl_vel_err = np.sqrt(np.sum((kl_est[:, :, 2:] - kl_unobs[:, :, 2:]) ** 2, axis=2))
            kl_em_vel_err = np.sqrt(np.sum((kl_EM_est[:, :, 2:] - kl_unobs[:, :, 2:]) ** 2, axis=2))
            kl_vel_diff_all[r_idx] = kl_vel_err - kl_em_vel_err
        except FileNotFoundError:
            print(f"警告：找不到KL算法在半径{radius}下的数据")
    
    # 创建位置误差比较图
    plt.figure(figsize=(12, 10))
    
    # 位置误差ECDF图
    plt.subplot(2, 1, 1)
    total_n = n_radi * n_ins * n_step
    errdf = []
    errdf.append(pd.DataFrame({'算法':['BCOT - EM',] * total_n, '误差差异': cot_poi_diff_all.reshape(-1)}))
    errdf.append(pd.DataFrame({'算法':['OT - EM',] * total_n, '误差差异': ot_poi_diff_all.reshape(-1)}))
    errdf.append(pd.DataFrame({'算法':['KL - EM',] * total_n, '误差差异': kl_poi_diff_all.reshape(-1)}))
    errdf = pd.concat(errdf)
    errdf = errdf.reset_index(drop=True)
    
    ax = sns.ecdfplot(data=errdf, x='误差差异', hue='算法')
    ax.set_xlim(-10, 6)
    plt.xlabel('位置误差差异', fontsize=14)
    plt.ylabel('比例', fontsize=14)
    plt.title('不同算法相对于EM算法的位置误差差异累积分布', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True)
    
    # 速度误差ECDF图
    plt.subplot(2, 1, 2)
    errdf1 = []
    errdf1.append(pd.DataFrame({'算法':['BCOT - EM',] * total_n, '误差差异': cot_vel_diff_all.reshape(-1)}))
    errdf1.append(pd.DataFrame({'算法':['OT - EM',] * total_n, '误差差异': ot_vel_diff_all.reshape(-1)}))
    errdf1.append(pd.DataFrame({'算法':['KL - EM',] * total_n, '误差差异': kl_vel_diff_all.reshape(-1)}))
    errdf1 = pd.concat(errdf1)
    errdf1 = errdf1.reset_index(drop=True)
    
    ax = sns.ecdfplot(data=errdf1, x='误差差异', hue='算法')
    ax.set_xlim(-10, 6)
    plt.xlabel('速度误差差异', fontsize=14)
    plt.ylabel('比例', fontsize=14)
    plt.title('不同算法相对于EM算法的速度误差差异累积分布', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('./logs/position_velocity_errors_comparison.pdf', format='pdf', dpi=1000, bbox_inches='tight', pad_inches=0.1)
    
    # 计算并打印各算法的平均误差和标准差
    print("\n位置误差统计信息:")
    print(f"{'算法':^10}|{'平均误差':^12}|{'标准差':^12}")
    print("-" * 36)
    
    cot_poi_mean = np.nanmean(cot_poi_diff_all)
    cot_poi_std = np.nanstd(cot_poi_diff_all)
    print(f"{'BCOT':^10}|{cot_poi_mean:^12.4f}|{cot_poi_std:^12.4f}")
    
    ot_poi_mean = np.nanmean(ot_poi_diff_all)
    ot_poi_std = np.nanstd(ot_poi_diff_all)
    print(f"{'OT':^10}|{ot_poi_mean:^12.4f}|{ot_poi_std:^12.4f}")
    
    kl_poi_mean = np.nanmean(kl_poi_diff_all)
    kl_poi_std = np.nanstd(kl_poi_diff_all)
    print(f"{'KL':^10}|{kl_poi_mean:^12.4f}|{kl_poi_std:^12.4f}")
    
    print("\n速度误差统计信息:")
    print(f"{'算法':^10}|{'平均误差':^12}|{'标准差':^12}")
    print("-" * 36)
    
    cot_vel_mean = np.nanmean(cot_vel_diff_all)
    cot_vel_std = np.nanstd(cot_vel_diff_all)
    print(f"{'BCOT':^10}|{cot_vel_mean:^12.4f}|{cot_vel_std:^12.4f}")
    
    ot_vel_mean = np.nanmean(ot_vel_diff_all)
    ot_vel_std = np.nanstd(ot_vel_diff_all)
    print(f"{'OT':^10}|{ot_vel_mean:^12.4f}|{ot_vel_std:^12.4f}")
    
    kl_vel_mean = np.nanmean(kl_vel_diff_all)
    kl_vel_std = np.nanstd(kl_vel_diff_all)
    print(f"{'KL':^10}|{kl_vel_mean:^12.4f}|{kl_vel_std:^12.4f}")
    
    print(f"\n{bcolors.GREEN}位置和速度误差比较图已保存到 ./logs/position_velocity_errors_comparison.pdf{bcolors.ENDC}")
def run_pairstrading(algo):
    """运行pairstrading.py应用配对交易策略"""
    print(f"\n{bcolors.BLUE}正在运行 {algo} 算法的配对交易策略...{bcolors.ENDC}")
    
    # 修改pairstrading.py中的ALGO_FLAG
    with open('pairstrading.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换算法标志
    content = content.replace("ALGO_FLAG = 'BCOT'", f"ALGO_FLAG = '{algo}'")
    content = content.replace("ALGO_FLAG = 'OT'", f"ALGO_FLAG = '{algo}'")
    content = content.replace("ALGO_FLAG = 'KL'", f"ALGO_FLAG = '{algo}'")
    content = content.replace("ALGO_FLAG = 'Nonrobust'", f"ALGO_FLAG = '{algo}'")
    
    # 写回文件
    with open('pairstrading.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    # 运行pairstrading.py
    start_time = time.time()
    subprocess.run(['python', 'pairstrading.py'], check=True)
    end_time = time.time()
    
    print(f"{bcolors.GREEN}{algo} 算法配对交易策略完成，耗时 {end_time - start_time:.2f} 秒{bcolors.ENDC}")

def analyze_tracking_results():
    """分析跟踪结果并生成比较图表"""
    print(f"\n{bcolors.BLUE}正在分析跟踪结果并生成比较图表...{bcolors.ENDC}")
    
    # 运行RMSE.py计算位置和速度误差
    print("运行RMSE.py计算BCOT和OT算法的位置和速度误差...")
    subprocess.run(['python', 'RMSE.py'], check=True)
    
    # 运行KL_RMSE.py计算KL散度
    print("运行KL_RMSE.py计算KL算法的位置和速度误差...")
    subprocess.run(['python', 'KL_RMSE.py'], check=True)
    
    print(f"{bcolors.GREEN}跟踪结果分析完成{bcolors.ENDC}")

def compare_sharpe_sortino():
    """比较不同算法的Sharpe和Sortino比率"""
    print(f"\n{bcolors.BLUE}正在比较不同算法的Sharpe和Sortino比率...{bcolors.ENDC}")
    
    # 加载不同算法的Sharpe和Sortino比率
    sharpe_data = {}
    sortino_data = {}
    
    for algo in ['BCOT', 'OT', 'KL']:
        try:
            with open(f'./logs/sharpe_{algo}.pickle', 'rb') as fp:
                sharpe_data[algo] = pickle.load(fp)
            
            with open(f'./logs/sortino_{algo}.pickle', 'rb') as fp:
                sortino_data[algo] = pickle.load(fp)
        except FileNotFoundError:
            print(f"警告：找不到 {algo} 算法的Sharpe或Sortino比率数据")
    
    # 创建比较图表
    if sharpe_data and sortino_data:
        plt.figure(figsize=(12, 6))
        
        # Sharpe比率图
        plt.subplot(1, 2, 1)
        for algo, data in sharpe_data.items():
            if algo == 'BCOT':
                radi_arr = np.linspace(0.1, 1.0, len(data))
                plt.plot(radi_arr, data, label=algo)
            else:
                radi_arr = np.linspace(0.1, 1.0, len(data))
                plt.plot(radi_arr, data, label=algo)
        
        plt.xlabel('半径')
        plt.ylabel('Sharpe比率')
        plt.title('不同算法的Sharpe比率比较')
        plt.legend()
        plt.grid(True)
        
        # Sortino比率图
        plt.subplot(1, 2, 2)
        for algo, data in sortino_data.items():
            if algo == 'BCOT':
                radi_arr = np.linspace(0.1, 1.0, len(data))
                plt.plot(radi_arr, data, label=algo)
            else:
                radi_arr = np.linspace(0.1, 1.0, len(data))
                plt.plot(radi_arr, data, label=algo)
        
        plt.xlabel('半径')
        plt.ylabel('Sortino比率')
        plt.title('不同算法的Sortino比率比较')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('./logs/sharpe_sortino_comparison.pdf', format='pdf', dpi=1000, bbox_inches='tight', pad_inches=0.1)
        print(f"{bcolors.GREEN}Sharpe和Sortino比率比较图已保存到 ./logs/sharpe_sortino_comparison.pdf{bcolors.ENDC}")
    else:
        print("没有足够的数据来创建Sharpe和Sortino比率比较图")

def create_comprehensive_report():
    """创建综合性能报告"""
    print(f"\n{bcolors.BLUE}正在创建综合性能报告...{bcolors.ENDC}")
    
    # 创建报告目录
    report_dir = "./logs/comprehensive_report"
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    
    # 创建综合报告文件
    with open(f"{report_dir}/performance_summary.txt", "w", encoding="utf-8") as f:
        f.write("卡尔曼滤波器算法性能比较综合报告\n")
        f.write("=================================\n\n")
        
        # 位置和速度误差比较
        f.write("1. 位置和速度误差比较\n")
        f.write("-------------------------\n")
        f.write("不同算法相对于EM算法的位置误差差异:\n")
        
        # 从RMSE.py和KL_RMSE.py的结果中提取数据
        radi_arr = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
        n_radi = 8
        
        f.write(f"{'半径':^10}|{'BCOT均值':^12}|{'BCOT标准差':^12}|{'OT均值':^12}|{'OT标准差':^12}|{'KL均值':^12}\n")
        f.write("-" * 60 + "\n")
        
        for r_idx in range(n_radi):
            radius = radi_arr[r_idx]
            
            # 加载BCOT数据
            cot_dir = f"./logs/BCOT_track_radi_{radius}"
            try:
                with open(f'{cot_dir}/unobs_state.pickle', 'rb') as fp:
                    unobs = pickle.load(fp)
                with open(f'{cot_dir}/est_state.pickle', 'rb') as fp:
                    cot_est = pickle.load(fp)
                with open(f'{cot_dir}/EM_est.pickle', 'rb') as fp:
                    EM_est = pickle.load(fp)
                    
                unobs = np.array(unobs)
                cot_est = np.array(cot_est)
                EM_est = np.array(EM_est)
                
                cot_poi_err = np.sqrt(np.sum((cot_est[:, :, :2] - unobs[:, :, :2]) ** 2, axis=2))
                em_poi_err = np.sqrt(np.sum((EM_est[:, :, :2] - unobs[:, :, :2]) ** 2, axis=2))
                cot_poi_diff = cot_poi_err - em_poi_err
                
                cot_mean = np.mean(cot_poi_diff)
                cot_std = np.std(cot_poi_diff)
            except FileNotFoundError:
                cot_mean = float('nan')
                cot_std = float('nan')
            
            # 加载OT数据
            ot_dir = f"./logs/OT_track_radi_{radius}"
            try:
                with open(f'{ot_dir}/unobs_state.pickle', 'rb') as fp:
                    ot_unobs = pickle.load(fp)
                with open(f'{ot_dir}/est_state.pickle', 'rb') as fp:
                    ot_est = pickle.load(fp)
                with open(f'{ot_dir}/EM_est.pickle', 'rb') as fp:
                    ot_EM_est = pickle.load(fp)
                    
                ot_unobs = np.array(ot_unobs)
                ot_est = np.array(ot_est)
                ot_EM_est = np.array(ot_EM_est)
                
                ot_poi_err = np.sqrt(np.sum((ot_est[:, :, :2] - ot_unobs[:, :, :2]) ** 2, axis=2))
                ot_em_poi_err = np.sqrt(np.sum((ot_EM_est[:, :, :2] - ot_unobs[:, :, :2]) ** 2, axis=2))
                ot_poi_diff = ot_poi_err - ot_em_poi_err
                
                ot_mean = np.mean(ot_poi_diff)
                ot_std = np.std(ot_poi_diff)
            except FileNotFoundError:
                ot_mean = float('nan')
                ot_std = float('nan')
            
            # 加载KL数据
            kl_dir = f"./logs/KL_track_radi_{radius}"
            try:
                with open(f'{kl_dir}/unobs_state.pickle', 'rb') as fp:
                    kl_unobs = pickle.load(fp)
                with open(f'{kl_dir}/est_state.pickle', 'rb') as fp:
                    kl_est = pickle.load(fp)
                with open(f'{kl_dir}/EM_est.pickle', 'rb') as fp:
                    kl_EM_est = pickle.load(fp)
                    
                kl_unobs = np.array(kl_unobs)
                kl_est = np.array(kl_est)
                kl_EM_est = np.array(kl_EM_est)
                
                kl_poi_err = np.sqrt(np.sum((kl_est[:, :, :2] - kl_unobs[:, :, :2]) ** 2, axis=2))
                kl_em_poi_err = np.sqrt(np.sum((kl_EM_est[:, :, :2] - kl_unobs[:, :, :2]) ** 2, axis=2))
                kl_poi_diff = kl_poi_err - kl_em_poi_err
                
                kl_mean = np.mean(kl_poi_diff)
            except FileNotFoundError:
                kl_mean = float('nan')
            
            f.write(f"{radius:^10.1f}|{cot_mean:^12.4f}|{cot_std:^12.4f}|{ot_mean:^12.4f}|{ot_std:^12.4f}|{kl_mean:^12.4f}\n")
        
        f.write("注：表格中的值是算法误差与EM算法误差的差值，负值表示算法优于EM算法\n\n")
        
        # Sharpe和Sortino比率比较
        f.write("2. Sharpe和Sortino比率比较\n")
        f.write("-------------------------\n")
        
        # 加载不同算法的Sharpe和Sortino比率
        sharpe_data = {}
        sortino_data = {}
        
        for algo in ['BCOT', 'OT', 'KL']:
            try:
                with open(f'./logs/sharpe_{algo}.pickle', 'rb') as fp:
                    sharpe_data[algo] = pickle.load(fp)
                
                with open(f'./logs/sortino_{algo}.pickle', 'rb') as fp:
                    sortino_data[algo] = pickle.load(fp)
            except FileNotFoundError:
                f.write(f"警告：找不到 {algo} 算法的Sharpe或Sortino比率数据\n")
        
        if sharpe_data:
            f.write("\nSharpe比率比较:\n")
            f.write(f"{'半径':^10}|")
            for algo in sharpe_data.keys():
                f.write(f"{algo:^12}|")
            f.write("\n")
            f.write("-" * (10 + 13 * len(sharpe_data)) + "\n")
            
            radi_arr = np.linspace(0.1, 1.0, len(next(iter(sharpe_data.values()))))
            for i, radius in enumerate(radi_arr):
                f.write(f"{radius:^10.2f}|")
                for algo, data in sharpe_data.items():
                    f.write(f"{data[i]:^12.4f}|")
                f.write("\n")
        
        if sortino_data:
            f.write("\nSortino比率比较:\n")
            f.write(f"{'半径':^10}|")
            for algo in sortino_data.keys():
                f.write(f"{algo:^12}|")
            f.write("\n")
            f.write("-" * (10 + 13 * len(sortino_data)) + "\n")
            
            radi_arr = np.linspace(0.1, 1.0, len(next(iter(sortino_data.values()))))
            for i, radius in enumerate(radi_arr):
                f.write(f"{radius:^10.2f}|")
                for algo, data in sortino_data.items():
                    f.write(f"{data[i]:^12.4f}|")
                f.write("\n")
        
        # 结论
        f.write("\n3. 结论\n")
        f.write("-------------------------\n")
        f.write("根据以上分析，我们可以得出以下结论：\n")
        f.write("1. 在位置和速度误差方面，")
        
        # 简单分析哪个算法表现最好
        try:
            cot_avg = np.nanmean([np.mean(cot_poi_diff) for cot_poi_diff in cot_poi_diff])
            ot_avg = np.nanmean([np.mean(ot_poi_diff) for ot_poi_diff in ot_poi_diff])
            kl_avg = np.nanmean([np.mean(kl_poi_diff) for kl_poi_diff in kl_poi_diff])
            
            best_algo = min([(cot_avg, "BCOT"), (ot_avg, "OT"), (kl_avg, "KL")], key=lambda x: x[0])[1]
            f.write(f"{best_algo}算法整体表现最好，误差相对于EM算法最小。\n")
        except:
            f.write("由于数据不完整，无法确定哪个算法在位置和速度误差方面表现最好。\n")
        
        # Sharpe和Sortino比率分析
        f.write("2. 在配对交易策略中，")
        try:
            sharpe_avg = {algo: np.mean(data) for algo, data in sharpe_data.items()}
            best_sharpe = max(sharpe_avg.items(), key=lambda x: x[1])[0]
            
            sortino_avg = {algo: np.mean(data) for algo, data in sortino_data.items()}
            best_sortino = max(sortino_avg.items(), key=lambda x: x[1])[0]
            
            if best_sharpe == best_sortino:
                f.write(f"{best_sharpe}算法在Sharpe比率和Sortino比率方面均表现最好。\n")
            else:
                f.write(f"{best_sharpe}算法在Sharpe比率方面表现最好，而{best_sortino}算法在Sortino比率方面表现最好。\n")
        except:
            f.write("由于数据不完整，无法确定哪个算法在配对交易策略中表现最好。\n")
        
        f.write("\n总体而言，不同算法在不同场景下各有优势，选择合适的算法应根据具体应用场景和性能需求来决定。")
    
    print(f"{bcolors.GREEN}综合性能报告已创建，保存在 {report_dir}/performance_summary.txt{bcolors.ENDC}")

def main():
    """主函数，运行所有算法并比较性能"""
    print(f"{bcolors.BOLD}{bcolors.BLUE}卡尔曼滤波器算法性能比较{bcolors.ENDC}")
    print("=================================\n")
    
    # 确保logs目录存在
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    
    # 运行所有算法的跟踪模拟
    #for algo in ['BCOT', 'OT', 'KL']:
        #run_tracking(algo)
    
    # 分析跟踪结果
    analyze_tracking_results()
    
    # 比较位置和速度误差
    compare_position_velocity_errors()
    
    # 运行所有算法的配对交易策略
    for algo in ['BCOT', 'OT', 'KL']:
        run_pairstrading(algo)
    
    # 比较Sharpe和Sortino比率
    compare_sharpe_sortino()
    
    # 创建综合性能报告
    create_comprehensive_report()
    
    print(f"\n{bcolors.GREEN}{bcolors.BOLD}所有算法比较完成！请查看logs目录下的结果文件和综合报告。{bcolors.ENDC}")

if __name__ == "__main__":
    main()
