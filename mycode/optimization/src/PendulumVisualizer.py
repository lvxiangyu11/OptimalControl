import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
plt.rcParams["font.sans-serif"]=["SimHei"] 
plt.rcParams["axes.unicode_minus"]=False

class PendulumVisualizer:
    def __init__(self, pendulum_length=1.0):
        self.pendulum_length = pendulum_length
        
        # 创建图形布局
        self.fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
        
        # 单摆可视化区域
        self.ax1 = plt.subplot(gs[0])
        self.ax1.set_xlim(-1.5, 1.5)
        self.ax1.set_ylim(-1.5, 1.5)
        self.ax1.set_aspect('equal')
        self.ax1.grid(True)
        self.ax1.set_title('单摆运动')
        
        # 状态绘制区域
        self.ax2 = plt.subplot(gs[1])
        self.ax2.grid(True)
        self.ax2.set_title('单摆状态')
        self.ax2.set_xlabel('时间 (s)')
        self.ax2.set_ylabel('角度 (rad)')
        
        # 创建单摆对象
        self.pendulum_line, = self.ax1.plot([], [], 'o-', lw=2, color='blue')
        self.time_template = '时间 = %.1fs'
        self.time_text = self.ax1.text(0.05, 0.9, '', transform=self.ax1.transAxes)
        
        # 创建状态图对象
        self.state_line, = self.ax2.plot([], [], lw=2, color='red')
        self.scatter_current = self.ax2.scatter([], [], color='blue', s=50, zorder=3)
        
        # 动画对象
        self.ani = None
    
    def get_pendulum_position(self, theta):
        """计算单摆末端的位置"""
        x = self.pendulum_length * np.sin(theta)
        y = -self.pendulum_length * np.cos(theta)
        return x, y
    
    def init_animation(self):
        """初始化动画"""
        self.pendulum_line.set_data([], [])
        self.time_text.set_text('')
        self.state_line.set_data([], [])
        self.scatter_current.set_offsets(np.array([0, 0]))
        return self.pendulum_line, self.time_text, self.state_line, self.scatter_current
    
    def update_animation(self, frame):
        """更新动画帧"""
        theta = self.x_hist[0, frame]
        x, y = self.get_pendulum_position(theta)
        
        # 更新单摆位置
        self.pendulum_line.set_data([0, x], [0, y])
        self.time_text.set_text(self.time_template % self.t_hist[frame])
        
        # 更新状态图
        self.state_line.set_data(self.t_hist[:frame+1], self.x_hist[0, :frame+1])
        self.scatter_current.set_offsets([self.t_hist[frame], self.x_hist[0, frame]])
        
        return self.pendulum_line, self.time_text, self.state_line, self.scatter_current
    
    def visualize(self, x_hist, t_hist):
        """创建并显示动画"""
        self.x_hist = x_hist
        self.t_hist = t_hist
        
        # 设置状态图的范围
        self.ax2.set_xlim(0, t_hist[-1])
        y_min = np.min(x_hist[0]) - 0.1
        y_max = np.max(x_hist[0]) + 0.1
        self.ax2.set_ylim(y_min, y_max)
        
        # 创建动画
        frames = len(t_hist)
        interval = (t_hist[1] - t_hist[0]) * 1000  # 转换为毫秒
        self.ani = FuncAnimation(self.fig, self.update_animation, frames=frames,
                                init_func=self.init_animation, blit=True, interval=interval)
        
        plt.tight_layout()
        plt.show()
