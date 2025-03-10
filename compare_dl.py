import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns
from dl_pairstrading import main as dl_main
from pairstrading import main as traditional_main
import sys
import warnings
from matplotlib import font_manager
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.")

# 设置支持中文的字体
font_path = 'C:/Windows/Fonts/msyh.ttc'  # 你可以根据你的系统调整字体路径
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
# 设置算法标志
algorithms = [
    'DL',       # 深度学习基础版本
    'DL_KL',    # 深度学习 + KL散度
    'DL_OT',    # 深度学习 + 最优传输
    'DL_BCOT',  # 深度学习 + 双因果最优传输
    #'KL',       # 传统KL散度
    #'OT',       # 传统最优传输
    #'BCOT'      # 传统双因果最优传输
]

# 鲁棒性半径范围
radii = [0.1, 0.3, 0.5, 0.7, 1.0]

# 结果存储
results = {
    'Algorithm': [],
    'Radius': [],
    'Sharpe': [],
    'Sortino': [],
    'Terminal_Wealth': []
}

def modify_algo_flag(file_path, algo):
    """修改文件中的ALGO_FLAG变量"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if 'ALGO_FLAG =' in line:
            lines[i] = f"ALGO_FLAG = '{algo}'  # 可选: 'DL', 'DL_KL', 'DL_OT', 'DL_BCOT', 'KL', 'OT', 'BCOT'\n"
            break
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def modify_radius(file_path, radius):
    """修改文件中的radius变量"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if 'radius =' in line and not 'search_num' in line:
            lines[i] = f"radius = {radius}\n"
            break
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def run_comparison():
    """运行所有算法和半径的比较"""
    # 确保日志目录存在
    os.makedirs('./logs/comparison', exist_ok=True)
    
    # 运行每个算法和半径组合
    for algo in algorithms:
        for radius in radii:
            print(f"\n运行算法: {algo}, 半径: {radius}")
            
            # 确定使用哪个主函数
            if algo in ['DL', 'DL_KL', 'DL_OT', 'DL_BCOT']:
                # 修改dl_pairstrading.py中的参数
                modify_algo_flag('d:\\pythonproject\\GaussianCOT_release\\dl_pairstrading.py', algo)
                modify_radius('d:\\pythonproject\\GaussianCOT_release\\dl_pairstrading.py', radius)
                
                # 运行深度学习版本
                sharpe, sortino, terminal_wealth = dl_main()
            else:
                # 修改pairstrading.py中的参数
                modify_algo_flag('d:\\pythonproject\\GaussianCOT_release\\pairstrading.py', algo)
                modify_radius('d:\\pythonproject\\GaussianCOT_release\\pairstrading.py', radius)
                
                # 运行传统版本
                # 注意：需要修改pairstrading.py的main函数以返回性能指标
                sharpe, sortino, terminal_wealth = traditional_main()
            
            # 存储结果
            results['Algorithm'].append(algo)
            results['Radius'].append(radius)
            results['Sharpe'].append(sharpe)
            results['Sortino'].append(sortino)
            results['Terminal_Wealth'].append(terminal_wealth)
    
    # 将结果转换为DataFrame
    results_df = pd.DataFrame(results)
    
    # 保存结果
    results_df.to_csv('./logs/comparison/all_results.csv', index=False)
    
    # 使用pickle保存结果
    with open('./logs/comparison/all_results.pickle', 'wb') as f:
        pickle.dump(results, f)
    
    return results_df

def plot_results(results_df):
    """绘制比较结果"""
    # 设置风格
    sns.set_theme(style="whitegrid")
    
    # 绘制Sharpe比率比较
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=results_df, x='Radius', y='Sharpe', hue='Algorithm', marker='o')
    plt.title('不同算法和半径的Sharpe比率比较', fontsize=16)
    plt.xlabel('鲁棒性半径', fontsize=14)
    plt.ylabel('Sharpe比率', fontsize=14)
    plt.grid(True)
    plt.savefig('./logs/comparison/sharpe_comparison.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig('./logs/comparison/sharpe_comparison.png', format='png', dpi=300, bbox_inches='tight')
    
    # 绘制Sortino比率比较
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=results_df, x='Radius', y='Sortino', hue='Algorithm', marker='o')
    plt.title('不同算法和半径的Sortino比率比较', fontsize=16)
    plt.xlabel('鲁棒性半径', fontsize=14)
    plt.ylabel('Sortino比率', fontsize=14)
    plt.grid(True)
    plt.savefig('./logs/comparison/sortino_comparison.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig('./logs/comparison/sortino_comparison.png', format='png', dpi=300, bbox_inches='tight')
    
    # 绘制终端财富比较
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=results_df, x='Radius', y='Terminal_Wealth', hue='Algorithm', marker='o')
    plt.title('不同算法和半径的终端财富比较', fontsize=16)
    plt.xlabel('鲁棒性半径', fontsize=14)
    plt.ylabel('终端财富', fontsize=14)
    plt.grid(True)
    plt.savefig('./logs/comparison/terminal_wealth_comparison.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig('./logs/comparison/terminal_wealth_comparison.png', format='png', dpi=300, bbox_inches='tight')
    
    # 绘制算法性能的箱线图
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=results_df, x='Algorithm', y='Sharpe')
    plt.title('各算法Sharpe比率分布', fontsize=16)
    plt.xlabel('算法', fontsize=14)
    plt.ylabel('Sharpe比率', fontsize=14)
    plt.grid(True)
    plt.savefig('./logs/comparison/sharpe_boxplot.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig('./logs/comparison/sharpe_boxplot.png', format='png', dpi=300, bbox_inches='tight')

def main():
    """主函数"""
    print("开始比较不同算法的性能...")
    
    # 检查是否已有结果
    if os.path.exists('./logs/comparison/all_results.pickle'):
        print("发现已有结果，加载中...")
        with open('./logs/comparison/all_results.pickle', 'rb') as f:
            results = pickle.load(f)
        results_df = pd.DataFrame(results)
    else:
        print("运行新的比较...")
        results_df = run_comparison()
    
    # 绘制结果
    print("绘制比较图表...")
    plot_results(results_df)
    
    # 打印摘要
    print("\n性能摘要:")
    summary = results_df.groupby('Algorithm').agg({
        'Sharpe': ['mean', 'std', 'max'],
        'Sortino': ['mean', 'std', 'max'],
        'Terminal_Wealth': ['mean', 'std', 'max']
    })
    print(summary)
    
    # 保存摘要
    summary.to_csv('./logs/comparison/summary.csv')
    
    print("\n比较完成！结果已保存到 ./logs/comparison/ 目录")

if __name__ == "__main__":
    main()