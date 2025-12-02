import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

class FlyLoRA_Rank2_Expert_Interactive:
    def __init__(self):
        # --- 1. 参数设置 ---
        self.m, self.n = 12, 12     # 权重矩阵维度 (为了显示清楚保持较小)
        self.total_rank = 16        # 总秩 r=16
        self.rank_per_expert = 2    # 每个专家包含的秩 (Rank=2)
        self.num_experts = self.total_rank // self.rank_per_expert # 总专家数 = 8
        self.k_experts = 4          # 激活的专家数量 = 4
        
        self.lr = 0.1               # 学习率
        self.total_steps = 200      # 模拟步数
        self.current_step = 0
        
        # --- 2. 初始化模型 ---
        np.random.seed(42)
        self.target_dW = np.random.randn(self.m, self.n)
        
        # [Standard LoRA] (Rank=16, Full Dense)
        self.lora_A = np.random.randn(self.total_rank, self.n) * 0.1
        self.lora_B = np.zeros((self.m, self.total_rank))
        
        # [FlyLoRA Variant] (Rank=16, Grouped Experts)
        # A: Frozen Random Projection
        self.fly_A = np.random.randn(self.total_rank, self.n) * 0.1
        # B: Sparse Update
        self.fly_B = np.zeros((self.m, self.total_rank))
        
        # 预计算训练过程
        self.history = []
        self.pre_calculate_training()
        
        # --- 3. 绘图初始化 ---
        self.setup_plot()
        self.update_view()
        
        print("交互界面已启动。请点击鼠标或按 '右方向键' 查看每一步更新。")
        plt.show()

    def pre_calculate_training(self):
        lora_A = self.lora_A.copy()
        lora_B = self.lora_B.copy()
        fly_A = self.fly_A.copy()
        fly_B = self.fly_B.copy()
        
        for step in range(self.total_steps):
            x = np.random.randn(self.n, 1)
            
            # --- Standard LoRA (Dense) ---
            curr_dW_lora = lora_B @ lora_A
            diff_lora = curr_dW_lora - self.target_dW
            grad_B_lora = diff_lora @ lora_A.T
            grad_A_lora = lora_B.T @ diff_lora
            lora_B -= self.lr * grad_B_lora
            lora_A -= self.lr * grad_A_lora
            
            # --- FlyLoRA (Rank-2 Experts) ---
            # 1. Projection
            y = fly_A @ x # shape (16, 1)
            
            # 2. Grouped Expert Selection
            # 将 16维 向量 reshape 成 (8个专家, 2维)
            y_grouped = y.reshape(self.num_experts, self.rank_per_expert)
            
            # 计算每个专家的"分数" (这里用 L1 Norm: 绝对值之和)
            expert_scores = np.sum(np.abs(y_grouped), axis=1) # shape (8,)
            
            # 选 Top-4 Experts
            top_k_expert_indices = np.argsort(expert_scores)[-self.k_experts:]
            
            # 3. Sparse Update
            # 计算全梯度 (假设全局反向传播)
            curr_dW_fly = fly_B @ fly_A
            diff_fly = curr_dW_fly - self.target_dW
            grad_B_full = diff_fly @ fly_A.T # shape (m, 16)
            
            # 创建梯度掩码 (Mask)
            grad_mask = np.zeros_like(grad_B_full)
            
            active_cols = []
            for exp_idx in top_k_expert_indices:
                # 专家 i 对应列 [2*i, 2*i+1]
                start_col = exp_idx * self.rank_per_expert
                end_col = start_col + self.rank_per_expert
                grad_mask[:, start_col:end_col] = 1
                active_cols.extend(range(start_col, end_col))
                
            # 仅更新被激活专家的列
            fly_B -= self.lr * (grad_B_full * grad_mask)
            
            self.history.append({
                'lora_B': lora_B.copy(),
                'fly_B': fly_B.copy(),
                'fly_activations': y.flatten(),
                'top_k_experts': top_k_expert_indices,
                'active_cols': active_cols, # 用于绘图高亮
                'step': step
            })

    def setup_plot(self):
        self.fig = plt.figure(figsize=(18, 9), facecolor='#f5f5f5')
        self.gs = GridSpec(2, 4, width_ratios=[0.8, 1.2, 0.3, 1.2], height_ratios=[1, 1])
        self.fig.suptitle(f"Standard LoRA vs. FlyLoRA (Rank={self.total_rank})", fontsize=16, fontweight='bold')
        
        self.cmap = 'RdBu_r'
        self.vmax = 1.0

        # --- Row 1: Standard LoRA ---
        ax_txt_lora = self.fig.add_subplot(self.gs[0, 0])
        ax_txt_lora.axis('off')
        ax_txt_lora.text(0.1, 0.6, f"Standard LoRA\n(Rank={self.total_rank})\n\nFully Dense Update", fontsize=14)
        
        self.ax_lora_B = self.fig.add_subplot(self.gs[0, 1])
        self.ax_lora_A = self.fig.add_subplot(self.gs[0, 3])
        self.ax_lora_B.set_title(f"Matrix B (12x{self.total_rank})", color='green', fontweight='bold')
        self.ax_lora_A.set_title(f"Matrix A ({self.total_rank}x12)", color='green', fontweight='bold')
        
        # --- Row 2: FlyLoRA ---
        ax_txt_fly = self.fig.add_subplot(self.gs[1, 0])
        ax_txt_fly.axis('off')
        ax_txt_fly.text(0.1, 0.6, 
                        f"FlyLoRA Variant\n(Total Rank={self.total_rank})\n\n"
                        f"Config:\n"
                        f"• 4 Active Experts\n"
                        f"• Rank-2 per Expert\n"
                        f"• A is Frozen", fontsize=14)
        
        self.ax_fly_B = self.fig.add_subplot(self.gs[1, 1])
        self.ax_fly_mid = self.fig.add_subplot(self.gs[1, 2])
        self.ax_fly_A = self.fig.add_subplot(self.gs[1, 3])
        
        self.ax_fly_B.set_title("Matrix B (Sparse Update)", color='green', fontweight='bold')
        self.ax_fly_mid.set_title("Expert\nSelect", fontsize=10)
        self.ax_fly_A.set_title("Matrix A (FROZEN)", color='red', fontweight='bold')
        
        # Initial Plots
        init = self.history[0]
        self.im_lora_B = self.ax_lora_B.imshow(init['lora_B'], cmap=self.cmap, vmin=-self.vmax, vmax=self.vmax, aspect='auto')
        self.im_lora_A = self.ax_lora_A.imshow(self.lora_A, cmap=self.cmap, vmin=-self.vmax, vmax=self.vmax, aspect='auto')
        
        self.im_fly_B = self.ax_fly_B.imshow(init['fly_B'], cmap=self.cmap, vmin=-self.vmax, vmax=self.vmax, aspect='auto')
        self.im_fly_A = self.ax_fly_A.imshow(self.fly_A, cmap='Greys', vmin=-self.vmax, vmax=self.vmax, aspect='auto')
        
        # Activations Strip
        self.im_fly_act = self.ax_fly_mid.imshow(np.zeros((self.total_rank, 1)), cmap='Reds', vmin=0, vmax=1, aspect='auto')
        self.ax_fly_mid.set_xticks([])
        self.ax_fly_mid.set_yticks(np.arange(0.5, self.total_rank, 2)) # Grid lines
        self.ax_fly_mid.grid(which='major', axis='y', color='black', linestyle='-', linewidth=0.5)
        
        self.txt_step = self.fig.text(0.5, 0.05, "", ha='center', fontsize=14, fontweight='bold')
        self.patches_list = []

        # Interactive events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    def update_view(self):
        if self.current_step >= len(self.history): self.current_step = len(self.history) - 1
        
        data = self.history[self.current_step]
        
        self.im_lora_B.set_data(data['lora_B'])
        self.im_fly_B.set_data(data['fly_B'])
        
        # Update Activations
        acts = np.abs(data['fly_activations']).reshape(-1, 1)
        if acts.max() > 0: acts /= acts.max()
        self.im_fly_act.set_data(acts)
        
        # Update Highlights
        [p.remove() for p in self.patches_list]
        self.patches_list = []
        
        # Draw Expert Boxes
        # 遍历所有 8 个专家
        for i in range(self.num_experts):
            is_active = i in data['top_k_experts']
            
            # y 坐标范围 (Rank 空间)
            y_start = i * self.rank_per_expert - 0.5
            height = self.rank_per_expert
            
            # x 坐标范围 (Matrix B 的列)
            x_start = i * self.rank_per_expert - 0.5
            width = self.rank_per_expert
            
            if is_active:
                # 1. 在激活条上画蓝框 (2格高)
                rect_act = Rectangle((-0.45, y_start), 0.9, height, 
                                   linewidth=2, edgecolor='blue', facecolor='none')
                self.ax_fly_mid.add_patch(rect_act)
                self.patches_list.append(rect_act)
                
                # 2. 在 Matrix B 底部画红块 (2格宽)
                rect_b = Rectangle((x_start, self.m - 0.5), width, 1.0,
                                 linewidth=0, facecolor='red', alpha=0.6, clip_on=False)
                self.ax_fly_B.add_patch(rect_b)
                self.patches_list.append(rect_b)
        
        self.txt_step.set_text(f"Step: {self.current_step} | Active Config: {self.k_experts} Experts × Rank-{self.rank_per_expert}")
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes: 
            self.current_step += 1
            self.update_view()
    
    def on_key(self, event):
        if event.key == 'right': self.current_step += 1
        elif event.key == 'left': self.current_step = max(0, self.current_step - 1)
        self.update_view()

if __name__ == "__main__":
    app = FlyLoRA_Rank2_Expert_Interactive()