import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_flylora_backward():
    fig, ax = plt.figure(figsize=(14, 8)), plt.gca()
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Define styles
    frozen_color = '#d0e0ff'  # Light blue
    trainable_color = '#ffe0d0' # Light orange
    active_color = '#ffaa00'   # Darker orange for active parts
    grad_color = '#ff4444'     # Red for gradients
    arrow_props = dict(facecolor=grad_color, arrowstyle='->', linewidth=1.5, edgecolor=grad_color)
    
    # 1. Output Gradient
    rect_out = patches.Rectangle((11.5, 1), 0.5, 6, linewidth=1, edgecolor='black', facecolor='white')
    ax.add_patch(rect_out)
    ax.text(11.75, 7.2, "Gradient $\\nabla L$\n(from Loss)", ha='center', va='bottom', color=grad_color)
    
    # Arrow Output -> B
    ax.annotate("", xy=(10, 4), xytext=(11.5, 4), arrowprops=arrow_props)
    
    # 2. Matrix B (Sparse Update)
    rect_B = patches.Rectangle((8, 1), 2, 6, linewidth=1, edgecolor='black', facecolor=trainable_color, alpha=0.3)
    ax.add_patch(rect_B)
    ax.text(9, 7.2, "Matrix $B$\n(Sparse Update)", ha='center', va='bottom')
    
    # Draw columns in B
    b_width = 2
    num_r = 5
    col_width = b_width / num_r
    k_indices = [0, 2] # Same as forward pass
    
    for i in range(num_r):
        x_pos = 8 + i * col_width
        ax.plot([x_pos, x_pos], [1, 7], 'k-', lw=0.5, alpha=0.3)
        
        if i in k_indices:
            # Active column getting update
            patch = patches.Rectangle((x_pos, 1), col_width, 6, color=active_color, alpha=0.8)
            ax.add_patch(patch)
            # Update symbol
            ax.text(x_pos + col_width/2, 4, "$\\leftarrow$ $\\nabla$", ha='center', va='center', color='white', fontsize=10, fontweight='bold')
            ax.text(x_pos + col_width/2, 0.5, "Upd", ha='center', fontsize=8, color='darkred', fontweight='bold')
        else:
            # Idle column
            ax.text(x_pos + col_width/2, 4, "0", ha='center', va='center', color='gray', fontsize=12)
            ax.text(x_pos + col_width/2, 0.5, "N.C.", ha='center', fontsize=8, color='grey')

    # Arrow B -> y
    ax.annotate("", xy=(6, 4), xytext=(8, 4), arrowprops=dict(facecolor='gray', arrowstyle='->', linestyle='dashed'))
    ax.text(7, 4.2, "Backprop", ha='center', color='gray', fontsize=8)

    # 3. Intermediate y gradient (Virtual)
    rect_y = patches.Rectangle((5.5, 2.5), 0.5, 3, linewidth=1, edgecolor='black', facecolor='white', alpha=0.5)
    ax.add_patch(rect_y)
    
    # Arrow y -> A
    ax.annotate("", xy=(4.5, 4), xytext=(5.5, 4), arrowprops=dict(facecolor='gray', arrowstyle='->', linestyle='dashed'))

    # 4. Matrix A (Frozen - Blocked Update)
    rect_A = patches.Rectangle((2.5, 2), 2, 4, linewidth=1, edgecolor='black', facecolor=frozen_color, alpha=0.5)
    ax.add_patch(rect_A)
    ax.text(3.5, 6.2, "Matrix $A$\n(Frozen)", ha='center', va='bottom')
    
    # Lock icon representation
    ax.text(3.5, 4, "No Update", ha='center', va='center', fontsize=12, fontweight='bold', color='#333333')

    # Arrow A -> x
    ax.annotate("", xy=(1.5, 4), xytext=(2.5, 4), arrowprops=dict(facecolor='gray', arrowstyle='->', linestyle='dashed'))

    # 5. Input Gradient
    rect_x = patches.Rectangle((1, 2), 0.5, 4, linewidth=1, edgecolor='black', facecolor='white', alpha=0.5)
    ax.add_patch(rect_x)
    ax.text(1.25, 6.2, "$\\nabla x$", ha='center', va='bottom', color='gray')

    # Title
    plt.title("FlyLoRA Backward Process (Gradient Flow)", fontsize=15, pad=20)
    
    # Save
    plt.savefig('flylora_backward.png', dpi=300, bbox_inches='tight')

draw_flylora_backward()