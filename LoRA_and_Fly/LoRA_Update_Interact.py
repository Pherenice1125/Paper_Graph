import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button

class LoraInteractive:
    def __init__(self):
        # 1. 模拟参数
        self.m, self.n = 10, 10
        self.r = 2
        self.steps = 100    # 总步数
        self.lr = 0.05
        self.current_step = 0
        
        # 2. 初始化数据
        np.random.seed(42)
        self.target_dW = np.random.randn(self.m, self.n)
        
        # LoRA 初始化
        self.A = np.random.randn(self.r, self.n) * 0.1
        self.B = np.zeros((self.m, self.r))
        
        # 预先计算所有步骤（为了交互流畅，我们先存下所有历史状态）
        self.history = self.simulate_training()
        
        # 3. 设置画布
        self.setup_plot()
        
        # 4. 连接交互事件
        # 鼠标点击任意位置触发
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        # 键盘右键触发
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        print("交互窗口已启动：\n请点击鼠标或按键盘 '右方向键' (->) 执行下一步。")
        plt.show()

    def simulate_training(self):
        """预先运行训练循环，保存每一步的状态"""
        history = []
        B = self.B.copy()
        A = self.A.copy()
        
        for i in range(self.steps + 1):
            dW = np.dot(B, A)
            diff = dW - self.target_dW
            loss = np.sum(diff**2)
            
            # 保存当前状态
            history.append({
                'B': B.copy(),
                'A': A.copy(),
                'dW': dW.copy(),
                'step': i,
                'loss': loss
            })
            
            # 梯度更新 (Gradient Descent)
            grad_dW = 2 * diff
            grad_B = np.dot(grad_dW, A.T)
            grad_A = np.dot(B.T, grad_dW)
            
            B -= self.lr * grad_B
            A -= self.lr * grad_A
            
        return history

    def setup_plot(self):
        self.fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 3, width_ratios=[1, 2, 2], height_ratios=[1, 1])
        
        # 布局
        self.ax_B = self.fig.add_subplot(gs[:, 0])
        self.ax_A = self.fig.add_subplot(gs[0, 1])
        self.ax_BA = self.fig.add_subplot(gs[1, 1])
        self.ax_T = self.fig.add_subplot(gs[1, 2])
        self.ax_Info = self.fig.add_subplot(gs[0, 2])
        self.ax_Info.axis('off')

        # 标题
        self.ax_B.set_title(f"Matrix B ({self.m}x{self.r})\nTrainable", fontsize=10)
        self.ax_A.set_title(f"Matrix A ({self.r}x{self.n})\nTrainable", fontsize=10)
        self.ax_BA.set_title("Result: $\Delta W = B \\times A$", fontsize=12, fontweight='bold')
        self.ax_T.set_title("Target Pattern (Goal)", fontsize=10, color='gray')

        # 绘图初始化
        vmax = np.max(np.abs(self.target_dW))
        cmap = 'RdBu_r'
        
        data_0 = self.history[0]
        self.im_B = self.ax_B.imshow(data_0['B'], cmap=cmap, vmin=-vmax, vmax=vmax)
        self.im_A = self.ax_A.imshow(data_0['A'], cmap=cmap, vmin=-vmax, vmax=vmax)
        self.im_BA = self.ax_BA.imshow(data_0['dW'], cmap=cmap, vmin=-vmax, vmax=vmax)
        self.im_T = self.ax_T.imshow(self.target_dW, cmap=cmap, vmin=-vmax, vmax=vmax)
        
        plt.colorbar(self.im_BA, ax=self.ax_BA, fraction=0.046, pad=0.04)

        # 文字
        self.txt_step = self.ax_Info.text(0.1, 0.7, f"Step: 0 / {self.steps}", fontsize=20, fontweight='bold')
        self.txt_loss = self.ax_Info.text(0.1, 0.5, f"Loss: {data_0['loss']:.4f}", fontsize=20, color='red')
        self.ax_Info.text(0.1, 0.3, "Click Mouse or Press 'Right Arrow'\nto verify next step update.", fontsize=12, color='blue')

        plt.tight_layout()

    def update_view(self):
        """根据 current_step 更新视图"""
        if self.current_step >= len(self.history):
            self.current_step = len(self.history) - 1
            return

        data = self.history[self.current_step]
        
        self.im_B.set_data(data['B'])
        self.im_A.set_data(data['A'])
        self.im_BA.set_data(data['dW'])
        
        self.txt_step.set_text(f"Step: {data['step']} / {self.steps}")
        self.txt_loss.set_text(f"Loss: {data['loss']:.4f}")
        
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        """鼠标点击事件处理"""
        # 只有点击在图表区域内才触发
        if event.inaxes is not None:
            self.current_step += 1
            self.update_view()

    def on_key(self, event):
        """键盘按键处理"""
        if event.key == 'right':
            self.current_step += 1
        elif event.key == 'left':
            self.current_step = max(0, self.current_step - 1) # 允许回退
        self.update_view()

# 运行交互程序
if __name__ == "__main__":
    app = LoraInteractive()