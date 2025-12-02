import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button

class LoRA_vs_FlyLoRA_Interactive:
    def __init__(self):
        # --- 1. 参数设置 ---
        self.m, self.n = 12, 12   # 权重矩阵维度
        self.r = 4                # Rank (秩)
        self.k = 2                # FlyLoRA 的 Top-k
        self.lr = 0.1             # 学习率
        self.total_steps = 200    # 模拟总步数
        self.current_step = 0
        
        # --- 2. 初始化模型 ---
        np.random.seed(42)
        # 目标：假设我们要学习一个随机的目标增量 delta W
        self.target_dW = np.random.randn(self.m, self.n)
        
        # [Standard LoRA]
        # A: 随机初始化, 可训练
        self.lora_A = np.random.randn(self.r, self.n) * 0.1
        # B: 零初始化, 可训练
        self.lora_B = np.zeros((self.m, self.r))
        
        # [FlyLoRA]
        # A: 随机初始化 (高斯), 冻结 (Frozen)
        self.fly_A = np.random.randn(self.r, self.n) * 0.1
        # B: 零初始化, 可训练
        self.fly_B = np.zeros((self.m, self.r))
        
        # 记录历史用于回放
        self.history = []
        self.pre_calculate_training()
        
        # --- 3. 绘图初始化 ---
        self.setup_plot()
        self.update_view()
        
        print("交互界面已启动。请点击鼠标或按 '右方向键' 查看每一步更新。")
        plt.show()

    def pre_calculate_training(self):
        """预先计算训练过程，保存每一步的状态"""
        lora_A = self.lora_A.copy()
        lora_B = self.lora_B.copy()
        fly_A = self.fly_A.copy() # FlyLoRA A is frozen, but we keep reference
        fly_B = self.fly_B.copy()
        
        for step in range(self.total_steps):
            # 随机生成一个输入向量 x (模拟 Batch Size = 1)
            x = np.random.randn(self.n, 1)
            
            # --- Standard LoRA Forward/Backward ---
            # Forward: dW = B @ A
            # Output for input x: out = B @ A @ x
            # Simplified Gradient Descent on Frobenious Norm ||BA - Target||^2
            # Gradients:
            # dL/dB = (BA - T) @ A.T
            # dL/dA = B.T @ (BA - T)
            
            current_dW_lora = lora_B @ lora_A
            diff_lora = current_dW_lora - self.target_dW
            
            grad_B_lora = diff_lora @ lora_A.T
            grad_A_lora = lora_B.T @ diff_lora
            
            # Update (Full Update)
            lora_B -= self.lr * grad_B_lora
            lora_A -= self.lr * grad_A_lora
            
            # --- FlyLoRA Forward/Backward ---
            # 1. Projection: y = A @ x
            y = fly_A @ x # shape (r, 1)
            
            # 2. Top-k Selection
            # 获取 y 中绝对值最大的 k 个索引
            top_k_indices = np.argsort(np.abs(y.flatten()))[-self.k:]
            
            # 创建掩码 (Mask)，只有 top-k 为 1，其余为 0
            mask = np.zeros_like(y)
            mask[top_k_indices] = 1
            
            # Effective activation: y_masked = y * mask
            # Output: out = B @ y_masked
            # 注意：在 FlyLoRA 中，只有被激活的 expert (column of B) 会参与计算和更新
            
            # Gradient:
            # 我们只更新 B 中对应的列。
            # dL/dB_cols = (Output - Target_x) * Activation_val
            # 这里为了简化可视化，我们仍然用全局误差近似，但只把梯度应用在选中的列上
            
            current_dW_fly = fly_B @ fly_A # 宏观上的当前权重
            # 计算针对当前 input x 的误差
            # pred = fly_B @ (fly_A @ x * mask)
            # target = self.target_dW @ x
            # err = pred - target
            
            # 为了可视化“矩阵更新”，我们使用全局矩阵误差，但只更新特定列
            # 这模拟了 FlyLoRA 在大量数据下的行为：每一步只动几列
            diff_fly = current_dW_fly - self.target_dW
            
            # 计算全梯度
            grad_B_full = diff_fly @ fly_A.T
            
            # Apply Sparse Update: 仅更新 top_k_indices 对应的列
            # 创建一个梯度掩码
            grad_B_mask = np.zeros_like(grad_B_full)
            grad_B_mask[:, top_k_indices] = 1
            
            # Update (Sparse Update), A is frozen
            fly_B -= self.lr * (grad_B_full * grad_B_mask) 
            
            # Save state
            self.history.append({
                'lora_A': lora_A.copy(),
                'lora_B': lora_B.copy(),
                'fly_B': fly_B.copy(),
                'fly_activations': y.flatten(),
                'top_k_idx': top_k_indices,
                'step': step,
                'lora_loss': np.sum(diff_lora**2),
                'fly_loss': np.sum(diff_fly**2)
            })

    def setup_plot(self):
        self.fig = plt.figure(figsize=(16, 9), facecolor='#f0f0f0')
        self.gs = GridSpec(2, 4, width_ratios=[1, 1, 0.2, 1], height_ratios=[1, 1])
        self.fig.suptitle("Standard LoRA vs. FlyLoRA (Interactive Training)", fontsize=16, fontweight='bold')
        
        # Color Map
        self.cmap = 'RdBu_r'
        self.vmax = 1.5

        # --- Row 1: Standard LoRA ---
        ax_lora_label = self.fig.add_subplot(self.gs[0, 0])
        ax_lora_label.axis('off')
        ax_lora_label.text(0.1, 0.5, "Standard LoRA\n(Rank=4)\n\nBoth A & B Update\nFully Dense", fontsize=14, color='#333')
        
        self.ax_lora_B = self.fig.add_subplot(self.gs[0, 1])
        self.ax_lora_A = self.fig.add_subplot(self.gs[0, 3])
        
        self.ax_lora_B.set_title("Matrix B (Trainable)", fontsize=12, color='green')
        self.ax_lora_A.set_title("Matrix A (Trainable)", fontsize=12, color='green')
        
        # --- Row 2: FlyLoRA ---
        ax_fly_label = self.fig.add_subplot(self.gs[1, 0])
        ax_fly_label.axis('off')
        ax_fly_label.text(0.1, 0.5, "FlyLoRA\n(Rank=4, k=2)\n\nA is Frozen\nB updates sparsely\n(Top-k Selection)", fontsize=14, color='#333')
        
        self.ax_fly_B = self.fig.add_subplot(self.gs[1, 1])
        self.ax_fly_mid = self.fig.add_subplot(self.gs[1, 2]) # 中间层激活显示
        self.ax_fly_A = self.fig.add_subplot(self.gs[1, 3])
        
        self.ax_fly_B.set_title("Matrix B (Sparse Update)", fontsize=12, color='green')
        self.ax_fly_mid.set_title("Top-k\nSelect", fontsize=10)
        self.ax_fly_A.set_title("Matrix A (FROZEN)", fontsize=12, color='red')
        
        # Initial Images
        init_data = self.history[0]
        
        # LoRA Plots
        self.im_lora_B = self.ax_lora_B.imshow(init_data['lora_B'], cmap=self.cmap, vmin=-self.vmax, vmax=self.vmax)
        self.im_lora_A = self.ax_lora_A.imshow(init_data['lora_A'], cmap=self.cmap, vmin=-self.vmax, vmax=self.vmax)
        
        # FlyLoRA Plots
        self.im_fly_B = self.ax_fly_B.imshow(init_data['fly_B'], cmap=self.cmap, vmin=-self.vmax, vmax=self.vmax)
        self.im_fly_A = self.ax_fly_A.imshow(self.fly_A, cmap='Greys', vmin=-self.vmax, vmax=self.vmax) # Grey for frozen
        
        # Activations (Vertical strip)
        act_data = np.zeros((self.r, 1))
        self.im_fly_act = self.ax_fly_mid.imshow(act_data, cmap='Reds', vmin=0, vmax=1)
        self.ax_fly_mid.set_xticks([])
        self.ax_fly_mid.set_yticks(range(self.r))
        
        # Markers for active columns
        self.active_indicators = []
        
        # Info Text
        self.txt_info = self.fig.text(0.5, 0.05, "", ha='center', fontsize=12, fontweight='bold')
        
        # Events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    def update_view(self):
        if self.current_step >= len(self.history):
            self.current_step = len(self.history) - 1
        
        data = self.history[self.current_step]
        
        # Update LoRA
        self.im_lora_B.set_data(data['lora_B'])
        self.im_lora_A.set_data(data['lora_A'])
        
        # Update FlyLoRA
        self.im_fly_B.set_data(data['fly_B'])
        
        # Update Activations Visualization
        # Show magnitude of activations
        acts = np.abs(data['fly_activations']).reshape(-1, 1)
        # Normalize for display
        if np.max(acts) > 0:
            acts = acts / np.max(acts)
        self.im_fly_act.set_data(acts)
        
        # Update Highlights (Active Experts)
        # remove old markers
        for marker in self.active_indicators:
            marker.remove()
        self.active_indicators = []
        
        # Add red box/arrow under active columns in B
        for idx in data['top_k_idx']:
            # 在 B 矩阵下方画一个小红色箭头或方块
            rect = plt.Rectangle((idx - 0.45, self.m - 0.5), 0.9, 1.0, 
                               fill=True, color='red', clip_on=False, alpha=0.6)
            self.ax_fly_B.add_patch(rect)
            self.active_indicators.append(rect)
            
            # 在 Activation 条上画框
            rect_act = plt.Rectangle((-0.45, idx - 0.45), 0.9, 0.9, 
                                   fill=False, edgecolor='blue', linewidth=2)
            self.ax_fly_mid.add_patch(rect_act)
            self.active_indicators.append(rect_act)

        # Update Text
        self.txt_info.set_text(f"Step: {self.current_step} | LoRA Loss: {data['lora_loss']:.4f} | FlyLoRA Loss: {data['fly_loss']:.4f}\n"
                               "Look at FlyLoRA Matrix B: Red blocks mark columns updated in this step (Top-2)")
        
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes is not None:
            self.current_step += 1
            self.update_view()

    def on_key(self, event):
        if event.key == 'right':
            self.current_step += 1
        elif event.key == 'left':
            self.current_step = max(0, self.current_step - 1)
        self.update_view()

if __name__ == "__main__":
    app = LoRA_vs_FlyLoRA_Interactive()