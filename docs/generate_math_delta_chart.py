import matplotlib.pyplot as plt
import numpy as np
import os

# cd docs && python generate_math_delta_chart.py

categories = [
    "Algebra", "Counting & Probability", "Geometry",
    "Intermediate Algebra", "Number Theory", "Prealgebra", "Precalculus"
]
vanilla = [62.01, 43.25, 32.99, 25.14, 42.22, 65.33, 24.54]
m_grpo = [89.39, 64.56, 56.99, 49.83, 70.56, 81.63, 51.28]
sb_grpo = [89.13, 67.09, 57.62, 54.71, 73.52, 82.43, 56.96]

def generate_chart(base_name, base_data, title, output_png):
    deltas = [sb - base for sb, base in zip(sb_grpo, base_data)]
    
    # Sort data in ascending order so that the largest value is at the top of the horizontal bar chart
    sorted_pairs = sorted(zip(categories, deltas), key=lambda x: x[1])
    sorted_cats = [x[0] for x in sorted_pairs]
    sorted_deltas = [x[1] for x in sorted_pairs]
    
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(12, 6.0))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    
    pos_color = "#2b5c8f"
    neg_color = "#c94c4c"
    colors = [pos_color if d > 0 else neg_color for d in sorted_deltas]
    
    y_pos = np.arange(len(sorted_cats))
    bars = ax.barh(
        y_pos, sorted_deltas, color=colors, height=0.75, alpha=0.9,
        edgecolor="#222222", linewidth=0.8
    )
    
    max_x = max(sorted_deltas) * 1.2
    min_x = min(min(sorted_deltas) * 1.2, -max_x * 0.02)
    
    for i, (bar, delta) in enumerate(zip(bars, sorted_deltas)):
        if delta >= 0:
            offset = max_x * 0.015
            ha = "left"
            x_pos = delta + offset
        else:
            offset = max_x * 0.015
            ha = "left"
            x_pos = offset
            
        color = pos_color if delta > 0 else neg_color
        sign = "+" if delta > 0 else ""
        ax.text(
            x_pos, bar.get_y() + bar.get_height() / 2,
            f"{sign}{delta:.2f}%", va="center", ha=ha,
            fontsize=13, fontweight="bold", color=color,
        )
        
    ax.axvline(0, color="black", linewidth=1.2, alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_cats, fontsize=14, fontweight="bold")
    ax.set_xlabel("Absolute Performance Gain (%)", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#333333")
    
    ax.set_xlim(min_x, max_x)
    
    ax.grid(axis="x", color="#e0e0e0", linestyle="--", alpha=0.7)
    ax.grid(axis="y", visible=False)
    
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {output_png}")

output_dir = r"d:\Projects\M-SB-GRPO\paper"
generate_chart("M-GRPO", m_grpo, "Performance Gain of SB-GRPO over M-GRPO on MATH Sub-categories", os.path.join(output_dir, "math_delta_chart_mgrpo.png"))
generate_chart("Vanilla GRPO", vanilla, "Performance Gain of SB-GRPO over Vanilla GRPO on MATH Sub-categories", os.path.join(output_dir, "math_delta_chart_vanilla.png"))
