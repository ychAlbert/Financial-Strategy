import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os


class KalmanFilterNet(nn.Module):
    """
    神经网络模型，用于学习最优的卡尔曼滤波参数
    该模型接收历史观测和状态估计，预测下一个状态
    """
    def __init__(self, obs_dim, state_dim, hidden_dim=64):
        super(KalmanFilterNet, self).__init__()
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        
        # 编码器：将观测和先前状态编码为隐藏表示
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 状态预测器：预测下一个状态的均值
        self.mean_predictor = nn.Linear(hidden_dim, state_dim)
        
        # 协方差预测器：预测下一个状态的协方差（对角元素）
        self.cov_predictor = nn.Sequential(
            nn.Linear(hidden_dim, state_dim),
            nn.Softplus()  # 确保协方差为正
        )
        
        # 卡尔曼增益预测器
        self.gain_predictor = nn.Linear(hidden_dim, state_dim * obs_dim)
    
    def forward(self, obs, prev_state):
        # 将观测和先前状态连接起来
        x = torch.cat([obs, prev_state], dim=1)
        
        # 编码
        h = self.encoder(x)
        
        # 预测均值和协方差
        mean = self.mean_predictor(h)
        cov_diag = self.cov_predictor(h)
        
        # 预测卡尔曼增益
        gain = self.gain_predictor(h).view(-1, self.state_dim, self.obs_dim)
        
        return mean, cov_diag, gain


class DeepKalmanFilter:
    """
    基于深度学习的卡尔曼滤波器实现
    结合了传统卡尔曼滤波的结构和深度学习的灵活性
    """
    def __init__(self, obs_dim, state_dim, hidden_dim=64, learning_rate=0.001):
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化神经网络模型
        self.model = KalmanFilterNet(obs_dim, state_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 用于数据标准化
        self.obs_scaler = StandardScaler()
        self.state_scaler = StandardScaler()
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
    
    def prepare_data(self, observations, states=None, train_ratio=0.8, batch_size=32):
        """
        准备训练数据
        
        Args:
            observations: 观测序列 [T, obs_dim]
            states: 真实状态序列（如果有）[T, state_dim]
            train_ratio: 训练集比例
            batch_size: 批量大小
        """
        # 标准化观测
        obs_normalized = self.obs_scaler.fit_transform(observations)
        obs_tensor = torch.FloatTensor(obs_normalized)
        
        # 如果有真实状态，则标准化
        if states is not None:
            states_normalized = self.state_scaler.fit_transform(states)
            states_tensor = torch.FloatTensor(states_normalized)
        else:
            # 如果没有真实状态，使用简单的线性回归初始化
            states_tensor = torch.zeros(len(observations), self.state_dim)
            # 这里可以添加初始化逻辑
        
        # 创建输入-输出对
        X, y = [], []
        for t in range(1, len(observations)):
            X.append(torch.cat([obs_tensor[t-1], states_tensor[t-1]], dim=0))
            y.append(states_tensor[t])
        
        X = torch.stack(X)
        y = torch.stack(y)
        
        # 分割训练集和验证集
        train_size = int(len(X) * train_ratio)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        return train_loader, val_loader
    
    def train(self, train_loader, val_loader, epochs=100, verbose=True):
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            verbose: 是否打印训练进度
        """
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                
                # 分离观测和状态
                obs = X[:, :self.obs_dim]
                prev_state = X[:, self.obs_dim:]
                
                # 前向传播
                self.optimizer.zero_grad()
                mean, cov_diag, _ = self.model(obs, prev_state)
                
                # 计算损失（均方误差）
                loss = criterion(mean, y)
                
                # 反向传播和优化
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * X.size(0)
            
            train_loss /= len(train_loader.dataset)
            self.train_losses.append(train_loss)
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    
                    # 分离观测和状态
                    obs = X[:, :self.obs_dim]
                    prev_state = X[:, self.obs_dim:]
                    
                    # 前向传播
                    mean, cov_diag, _ = self.model(obs, prev_state)
                    
                    # 计算损失
                    loss = criterion(mean, y)
                    val_loss += loss.item() * X.size(0)
            
            val_loss /= len(val_loader.dataset)
            self.val_losses.append(val_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    def predict(self, observations, initial_state=None):
        """
        使用训练好的模型进行预测
        
        Args:
            observations: 观测序列 [T, obs_dim]
            initial_state: 初始状态（如果没有，使用零向量）
        
        Returns:
            predicted_states: 预测的状态序列 [T, state_dim]
            predicted_covs: 预测的协方差序列 [T, state_dim]
        """
        self.model.eval()
        
        # 标准化观测
        obs_normalized = self.obs_scaler.transform(observations)
        obs_tensor = torch.FloatTensor(obs_normalized).to(self.device)
        
        # 初始化状态
        if initial_state is None:
            current_state = torch.zeros(1, self.state_dim).to(self.device)
        else:
            current_state = torch.FloatTensor(self.state_scaler.transform([initial_state])).to(self.device)
        
        predicted_states = []
        predicted_covs = []
        
        with torch.no_grad():
            for t in range(len(observations)):
                # 获取当前观测
                current_obs = obs_tensor[t:t+1]
                
                # 预测下一个状态
                mean, cov_diag, gain = self.model(current_obs, current_state)
                
                # 更新当前状态
                current_state = mean
                
                # 保存预测结果
                predicted_states.append(mean.cpu().numpy())
                predicted_covs.append(cov_diag.cpu().numpy())
        
        # 反标准化预测结果
        predicted_states = self.state_scaler.inverse_transform(np.vstack(predicted_states))
        
        return predicted_states, np.vstack(predicted_covs)
    
    def save_model(self, path):
        """
        保存模型
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'obs_scaler': self.obs_scaler,
            'state_scaler': self.state_scaler,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)
    
    def load_model(self, path):
        """
        加载模型
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.obs_scaler = checkpoint['obs_scaler']
        self.state_scaler = checkpoint['state_scaler']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
    
    def plot_training_history(self):
        """
        绘制训练历史
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        plt.show()


def dl_optimize(m_dim, n_dim, radius, A, Bp, C, Dp, Sigp, current_obs, pred_mean, model_path=None, algo='DL'):
    """
    使用深度学习模型进行状态估计优化
    
    Args:
        m_dim: 观测维度
        n_dim: 状态维度
        radius: 鲁棒性半径
        A: 状态转移矩阵
        Bp: 参考模型的过程噪声协方差
        C: 观测矩阵
        Dp: 参考模型的观测噪声协方差
        Sigp: 参考模型的状态协方差
        current_obs: 当前观测
        pred_mean: 预测均值
        model_path: 模型路径（如果为None，则使用默认路径）
        algo: 算法类型（'DL'或'DL_KL'或'DL_OT'）
    
    Returns:
        update_mean: 更新后的状态均值
        update_cov: 更新后的状态协方差
    """
    # 默认模型路径
    if model_path is None:
        model_path = f"./models/dl_kalman_{algo}.pt"
    
    # 检查模型是否存在
    if not os.path.exists(model_path):
        # 如果模型不存在，使用传统方法
        from utils import optimize
        print(f"深度学习模型不存在，使用传统{algo.replace('DL_', '')}方法")
        if algo == 'DL':
            return optimize(m_dim, n_dim, radius, A, Bp, C, Dp, Sigp, current_obs, pred_mean, 20, 'BCOT')
        elif algo == 'DL_KL':
            return optimize(m_dim, n_dim, radius, A, Bp, C, Dp, Sigp, current_obs, pred_mean, 2, 'KL')
        elif algo == 'DL_OT':
            return optimize(m_dim, n_dim, radius, A, Bp, C, Dp, Sigp, current_obs, pred_mean, 20, 'OT')
    
    # 创建深度卡尔曼滤波器
    dkf = DeepKalmanFilter(m_dim, n_dim)
    
    # 加载模型
    dkf.load_model(model_path)
    
    # 准备输入数据
    current_obs_np = current_obs.reshape(1, -1)
    pred_mean_np = pred_mean.reshape(1, -1)
    
    # 使用模型预测
    with torch.no_grad():
        # 标准化输入
        obs_normalized = dkf.obs_scaler.transform(current_obs_np)
        state_normalized = dkf.state_scaler.transform(pred_mean_np)
        
        # 转换为张量
        obs_tensor = torch.FloatTensor(obs_normalized).to(dkf.device)
        state_tensor = torch.FloatTensor(state_normalized).to(dkf.device)
        
        # 预测
        mean, cov_diag, gain = dkf.model(obs_tensor, state_tensor)
        
        # 反标准化
        update_mean = dkf.state_scaler.inverse_transform(mean.cpu().numpy())
        
        # 构建完整协方差矩阵（从对角元素）
        cov_np = np.diag(cov_diag.cpu().numpy()[0])
        
        # 根据算法类型调整协方差
        if algo == 'DL_KL' or algo == 'DL_OT':
            # 使用参考模型的结构，但用深度学习预测的值
            P = A @ Sigp @ A.T - gain.cpu().numpy()[0] @ Dp @ gain.cpu().numpy()[0].T + Bp
            update_cov = cov_np @ P @ cov_np.T
        else:
            # 直接使用预测的协方差
            update_cov = cov_np
    
    return update_mean.reshape(-1, 1), update_cov


def train_dl_model(observations, states, m_dim, n_dim, hidden_dim=64, epochs=100, batch_size=32, save_path=None):
    """
    训练深度学习模型
    
    Args:
        observations: 观测序列 [T, m_dim]
        states: 状态序列 [T, n_dim]
        m_dim: 观测维度
        n_dim: 状态维度
        hidden_dim: 隐藏层维度
        epochs: 训练轮数
        batch_size: 批量大小
        save_path: 保存路径
    
    Returns:
        dkf: 训练好的深度卡尔曼滤波器
    """
    # 创建深度卡尔曼滤波器
    dkf = DeepKalmanFilter(m_dim, n_dim, hidden_dim)
    
    # 准备数据
    train_loader, val_loader = dkf.prepare_data(observations, states, batch_size=batch_size)
    
    # 训练模型
    dkf.train(train_loader, val_loader, epochs=epochs)
    
    # 保存模型
    if save_path is not None:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        dkf.save_model(save_path)
    
    return dkf


class DLKalmanFilter:
    """
    深度学习增强的卡尔曼滤波器
    结合了传统卡尔曼滤波和深度学习的优点
    """
    def __init__(self, m_dim, n_dim, radius, A, C, model_path=None, algo='DL'):
        self.m_dim = m_dim
        self.n_dim = n_dim
        self.radius = radius
        self.A = A
        self.C = C
        self.algo = algo
        self.model_path = model_path if model_path else f"./models/dl_kalman_{algo}.pt"
        
        # 初始化参考模型参数
        self.Bp = np.eye(n_dim)
        self.Dp = np.eye(m_dim)
        self.Sigp = np.eye(n_dim)
        
        # 初始化状态
        self.current_mean = np.zeros((n_dim, 1))
        self.current_cov = np.eye(n_dim)
        
        # 检查模型是否存在
        if not os.path.exists(self.model_path):
            print(f"警告：模型文件 {self.model_path} 不存在，将使用传统方法")
    
    def update(self, observation, pred_mean=None):
        """
        更新状态估计
        
        Args:
            observation: 当前观测 [m_dim, 1]
            pred_mean: 预测均值（如果为None，使用A@current_mean）
        
        Returns:
            updated_mean: 更新后的状态均值
            updated_cov: 更新后的状态协方差
        """
        if pred_mean is None:
            pred_mean = self.A @ self.current_mean
        
        # 使用深度学习模型进行优化
        updated_mean, updated_cov = dl_optimize(
            self.m_dim, self.n_dim, self.radius,
            self.A, self.Bp, self.C, self.Dp, self.Sigp,
            observation, pred_mean, self.model_path, self.algo
        )
        
        # 更新当前状态
        self.current_mean = updated_mean
        self.current_cov = updated_cov
        
        return updated_mean, updated_cov
    
    def filter(self, observations):
        """
        对一系列观测进行滤波
        
        Args:
            observations: 观测序列 [T, m_dim]
        
        Returns:
            filtered_states: 滤波后的状态序列 [T, n_dim]
            filtered_covs: 滤波后的协方差序列 [T, n_dim, n_dim]
        """
        T = len(observations)
        filtered_states = np.zeros((T, self.n_dim))
        filtered_covs = np.zeros((T, self.n_dim, self.n_dim))
        
        for t in range(T):
            # 获取当前观测
            current_obs = observations[t].reshape(-1, 1)
            
            # 预测
            pred_mean = self.A @ self.current_mean
            
            # 更新
            updated_mean, updated_cov = self.update(current_obs, pred_mean)
            
            # 保存结果
            filtered_states[t] = updated_mean.reshape(-1)
            filtered_covs[t] = updated_cov
        
        return filtered_states, filtered_covs