import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

# 1. 设置模拟参数
m, n = 10, 10  # 原始权重矩阵维度 (10x10)
r = 2          # LoRA 的秩 (Rank = 2)
steps = 60     # 训练步数 (动画帧数)
lr = 0.05      # 学习率

# 2. 初始化数据
np.random.seed(42)
# 假设这是我们需要拟合的目标更新量 (Target Delta W)
target_dW = np.random.randn(m, n)

# LoRA 初始化标准：
# A 矩阵使用随机高斯初始化
A = np.random.randn(r, n) * 0.1
# B 矩阵初始化为全 0
B = np.zeros((m, r))

# 3. 训练循环 (收集每一帧的数据)
history = []
for i in range(steps):
    # 前向传播: 计算当前的 delta W
    dW = np.dot(B, A)
    
    # 计算损失 (MSE Loss)
    diff = dW - target_dW
    loss = np.sum(diff**2)
    
    # 反向传播 (计算梯度)
    # dL/dW = 2 * (BA - Target)
    grad_dW = 2 * diff
    # dL/dB = dL/dW * A^T
    grad_B = np.dot(grad_dW, A.T)
    # dL/dA = B^T * dL/dW
    grad_A = np.dot(B.T, grad_dW)
    
    # 更新权重 (梯度下降)
    B -= lr * grad_B
    A -= lr * grad_A
    
    # 保存这一帧的状态
    history.append({
        'B': B.copy(),
        'A': A.copy(),
        'dW': dW.copy(),
        'step': i,
        'loss': loss
    })

# 4. 设置可视化布局
fig = plt.figure(figsize=(12, 7))
# 使用 GridSpec 进行复杂的子图布局
gs = GridSpec(2, 3, width_ratios=[1, 2, 2], height_ratios=[1, 1])

# 定义子图位置
ax_B = fig.add_subplot(gs[:, 0])      # 左侧: B 矩阵 (瘦高)
ax_A = fig.add_subplot(gs[0, 1])      # 中上: A 矩阵 (扁平)
ax_BA = fig.add_subplot(gs[1, 1])     # 中下: BxA 结果 (正方形)
ax_T = fig.add_subplot(gs[1, 2])      # 右下: 目标矩阵 (参考)
ax_Info = fig.add_subplot(gs[0, 2])   # 右上: 文字信息
ax_Info.axis('off') # 隐藏坐标轴

# 设置标题
ax_B.set_title(f"Matrix B ({m}x{r})\nInit: Zeros", fontsize=11)
ax_A.set_title(f"Matrix A ({r}x{n})\nInit: Random", fontsize=11)
ax_BA.set_title(f"Current Update $\Delta W = B \\times A$", fontsize=11, fontweight='bold')
ax_T.set_title(f"Target Update Pattern", fontsize=11, color='gray')

# 统一颜色映射范围，便于观察数值大小
vmax = np.max(np.abs(target_dW))
cmap = 'RdBu_r' # 红蓝配色，0为白色

# 初始化图像对象
im_B = ax_B.imshow(history[0]['B'], cmap=cmap, vmin=-vmax, vmax=vmax)
im_A = ax_A.imshow(history[0]['A'], cmap=cmap, vmin=-vmax, vmax=vmax)
im_BA = ax_BA.imshow(history[0]['dW'], cmap=cmap, vmin=-vmax, vmax=vmax)
im_T = ax_T.imshow(target_dW, cmap=cmap, vmin=-vmax, vmax=vmax)

# 添加 Colorbar (仅给 BA 加一个作为参考)
plt.colorbar(im_BA, ax=ax_BA, fraction=0.046, pad=0.04)

# 动态文本对象
txt_step = ax_Info.text(0.1, 0.6, "", fontsize=16, fontweight='bold')
txt_loss = ax_Info.text(0.1, 0.4, "", fontsize=16, color='red')
ax_Info.text(0.1, 0.2, "Simulating Backpropagation...", fontsize=10, color='gray')

plt.tight_layout()

# 5. 动画更新函数
def update(frame):
    data = history[frame]
    
    # 更新矩阵图像数据
    im_B.set_data(data['B'])
    im_A.set_data(data['A'])
    im_BA.set_data(data['dW'])
    
    # 更新文字
    txt_step.set_text(f"Step: {data['step']}")
    txt_loss.set_text(f"Loss: {data['loss']:.4f}")
    
    return im_B, im_A, im_BA, txt_step, txt_loss

# 创建动画
ani = animation.FuncAnimation(fig, update, frames=len(history), interval=500, blit=True)

# 保存为 GIF (需要安装 pillow 或 imagemagick)
ani.save('lora_training_slow.gif', writer='pillow', fps=10)

print("动画已生成并保存为 lora_training_slow.gif")
plt.show()