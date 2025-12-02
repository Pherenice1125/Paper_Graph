import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_flylora_diagram():
    fig, ax = plt.figure(figsize=(14, 8)), plt.gca()
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Define styles
    frozen_color = '#d0e0ff'  # Light blue
    trainable_color = '#ffe0d0' # Light orange
    active_color = '#ffaa00'   # Darker orange for active parts
    arrow_props = dict(facecolor='black', arrowstyle='->', linewidth=1.5)
    
    # 1. Input x
    rect_x = patches.Rectangle((1, 2), 0.5, 4, linewidth=1, edgecolor='black', facecolor='white')
    ax.add_patch(rect_x)
    ax.text(1.25, 6.2, "Input $x$\n($n \\times 1$)", ha='center', va='bottom')
    
    # Arrow x -> A
    ax.annotate("", xy=(2.5, 4), xytext=(1.5, 4), arrowprops=arrow_props)
    
    # 2. Frozen Matrix A
    rect_A = patches.Rectangle((2.5, 2), 2, 4, linewidth=1, edgecolor='black', facecolor=frozen_color, alpha=0.5)
    ax.add_patch(rect_A)
    ax.text(3.5, 6.2, "Matrix $A$\n(Frozen, Random)\n($r \\times n$)", ha='center', va='bottom')
    # Add some "sparse" dots
    import numpy as np
    np.random.seed(42)
    for _ in range(15):
        ax.plot(2.5 + np.random.rand()*2, 2 + np.random.rand()*4, 'b.', markersize=2)
    
    # Arrow A -> y
    ax.annotate("", xy=(5.5, 4), xytext=(4.5, 4), arrowprops=arrow_props)
    
    # 3. Intermediate y = Ax
    rect_y = patches.Rectangle((5.5, 2.5), 0.5, 3, linewidth=1, edgecolor='black', facecolor='white')
    ax.add_patch(rect_y)
    ax.text(5.75, 5.7, "Vector $y = Ax$\n($r \\times 1$)", ha='center', va='bottom')
    
    # Highlight Top-k indices in y
    # Let's assume indices 1 and 3 (from top, 0-indexed) are top-k
    k_indices = [0, 2] 
    y_height = 3
    num_r = 5
    step = y_height / num_r
    for i in range(num_r):
        y_pos = 2.5 + y_height - (i + 1) * step
        if i in k_indices:
            patch = patches.Rectangle((5.5, y_pos), 0.5, step, color=active_color, alpha=0.7)
            ax.add_patch(patch)
            # Arrow from y to Top-k decision
            # ax.annotate("", xy=(7, y_pos + step/2), xytext=(6, y_pos + step/2), arrowprops=dict(facecolor='red', arrowstyle='->'))

    ax.text(5.75, 1.5, "Top-$k$ Selection", ha='center', color='red', fontweight='bold')
    
    # Bracket or lines mapping to B
    ax.annotate("", xy=(8, 4), xytext=(6, 4), arrowprops=arrow_props)
    
    # 4. Matrix B
    rect_B = patches.Rectangle((8, 1), 2, 6, linewidth=1, edgecolor='black', facecolor=trainable_color, alpha=0.3)
    ax.add_patch(rect_B)
    ax.text(9, 7.2, "Matrix $B$\n(Trainable)\n($m \\times r$)", ha='center', va='bottom')
    
    # Draw columns in B corresponding to y indices
    b_width = 2
    col_width = b_width / num_r
    for i in range(num_r):
        x_pos = 8 + i * col_width
        # Draw vertical lines separating columns
        ax.plot([x_pos, x_pos], [1, 7], 'k-', lw=0.5, alpha=0.3)
        
        if i in k_indices:
            # Highlight active columns
            patch = patches.Rectangle((x_pos, 1), col_width, 6, color=active_color, alpha=0.8)
            ax.add_patch(patch)
            ax.text(x_pos + col_width/2, 0.8, "Active", ha='center', fontsize=8, color='darkred')
        else:
            ax.text(x_pos + col_width/2, 0.8, "Idle", ha='center', fontsize=8, color='grey')

    # 5. Output
    ax.annotate("", xy=(11.5, 4), xytext=(10, 4), arrowprops=arrow_props)
    rect_out = patches.Rectangle((11.5, 1), 0.5, 6, linewidth=1, edgecolor='black', facecolor='white')
    ax.add_patch(rect_out)
    ax.text(11.75, 7.2, "Output\n$\\Delta W x$\n($m \\times 1$)", ha='center', va='bottom')

    # Add explanatory text
    plt.title("FlyLoRA Forward Process Visualization", fontsize=15, pad=20)
    
    # Legend-like explanation at the bottom
    # plt.figtext(0.5, 0.05, "1. Input projects via Frozen A -> y\n2. Top-k values in y are identified\n3. Only corresponding columns in B are activated & trained", ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

    plt.savefig('flylora_diagram.png', dpi=300, bbox_inches='tight')

draw_flylora_diagram()