import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set paths
csv_path = r"d:\Final Testing\eval\results\GPU_NVIDIA_A100-SXM4-40GB\reports\final\speed_accuracy_comparison.csv"
output_dir = r"d:\Final Testing\eval\results\GPU_NVIDIA_A100-SXM4-40GB\reports\final\plots"

# Read data
df = pd.read_csv(csv_path)

# Clean up model names for better display
df['model'] = df['model'].str.replace('final_', 'Final ', regex=False)
df['model'] = df['model'].str.replace('pure_coordgn', 'Pure CoordGn', regex=False)
df['model'] = df['model'].str.replace('pure_spd', 'Pure SPD', regex=False)

# Reorder based on user requested 1-based indices: 1, 2, 4, 9, 10, 7, and rest
# 0-based indices: 0, 1, 3, 8, 9, 6, 2, 4, 5, 7
custom_order_indices = [0, 1, 3, 8, 9, 6, 2, 4, 5, 7]
df = df.iloc[custom_order_indices].reset_index(drop=True)

plt.rcParams.update({'font.size': 12})
plt.style.use('ggplot')

# 1. Metrics Diagram (mAP_50, mAP_50_95, mAP_small)
fig, ax = plt.subplots(figsize=(14, 8))
models = df['model'].tolist()
x = np.arange(len(models))
width = 0.25

rects1 = ax.bar(x - width, df['mAP_50'], width, label='mAP_50', color='#1f77b4')
rects2 = ax.bar(x, df['mAP_50_95'], width, label='mAP_50_95', color='#ff7f0e')
rects3 = ax.bar(x + width, df['mAP_small'], width, label='mAP_small', color='#2ca02c')

# Add labels
ax.set_ylabel('mAP Score', fontsize=14)
ax.set_title('Model Comparison: mAP Metrics', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend()

# Add value labels
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('%.3f' % height,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=90, size=9)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.ylim(0, max(df['mAP_50']) * 1.3)
fig.tight_layout()

metrics_out = os.path.join(output_dir, 'metrics_diagram.png')
plt.savefig(metrics_out, dpi=300, bbox_inches='tight')
plt.close()

# 2. FPS Comparison Diagram
fig, ax = plt.subplots(figsize=(12, 7))

# Use the custom order
sorted_models = df['model'].tolist()
sorted_fps = df['fps'].tolist()

y_pos = np.arange(len(sorted_models))

# Create colormap based on FPS value
norm = plt.Normalize(min(sorted_fps), max(sorted_fps))
colors = plt.cm.coolwarm(norm(sorted_fps))

rects = ax.bar(y_pos, sorted_fps, align='center', color=colors)

ax.set_xticks(y_pos)
ax.set_xticklabels(sorted_models, rotation=45, ha='right')
ax.set_ylabel('Frames Per Second (FPS)', fontsize=14)
ax.set_title('FPS Comparison Across Models (NVIDIA A100)', fontsize=16, fontweight='bold')

def autolabel_fps(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('%.1f' % height,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', size=10)

autolabel_fps(rects)

plt.ylim(0, max(sorted_fps) * 1.15)
fig.tight_layout()

fps_out = os.path.join(output_dir, 'fps_comparison_diagram.png')
plt.savefig(fps_out, dpi=300, bbox_inches='tight')
plt.close()

# 3. mAP_small Comparison Diagram
fig, ax = plt.subplots(figsize=(12, 7))

map_small = df['mAP_small'].tolist()

# Use a single color for all bars
rects_m = ax.bar(y_pos, map_small, align='center', color='#2ca02c')

ax.set_xticks(y_pos)
ax.set_xticklabels(sorted_models, rotation=45, ha='right')
ax.set_ylabel('mAP_small Score', fontsize=14)
ax.set_title('mAP_small Comparison Across Models', fontsize=16, fontweight='bold')

def autolabel_map(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('%.3f' % height,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', size=10)

autolabel_map(rects_m)

plt.ylim(0, max(map_small) * 1.15)
fig.tight_layout()

map_small_out = os.path.join(output_dir, 'map_small_diagram.png')
plt.savefig(map_small_out, dpi=300, bbox_inches='tight')
plt.close()

# 4. mAP_50 Comparison Diagram
fig, ax = plt.subplots(figsize=(12, 7))

map_50 = df['mAP_50'].tolist()

# Use a single color for all bars
rects_50 = ax.bar(y_pos, map_50, align='center', color='#1f77b4')

ax.set_xticks(y_pos)
ax.set_xticklabels(sorted_models, rotation=45, ha='right')
ax.set_ylabel('mAP_50 Score', fontsize=14)
ax.set_title('mAP_50 Comparison Across Models', fontsize=16, fontweight='bold')

def autolabel_map50(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('%.3f' % height,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', size=10)

autolabel_map50(rects_50)

plt.ylim(0, max(map_50) * 1.15)
fig.tight_layout()

map_50_out = os.path.join(output_dir, 'map_50_diagram.png')
plt.savefig(map_50_out, dpi=300, bbox_inches='tight')
plt.close()

# 5. GFLOPs Comparison Diagram
fig, ax = plt.subplots(figsize=(12, 7))

gflops = df['GFLOPs'].tolist()

# Use a single color for all bars (purple)
rects_gflops = ax.bar(y_pos, gflops, align='center', color='#9467bd')

ax.set_xticks(y_pos)
ax.set_xticklabels(sorted_models, rotation=45, ha='right')
ax.set_ylabel('GFLOPs', fontsize=14)
ax.set_title('Computational Complexity (GFLOPs) Across Models', fontsize=16, fontweight='bold')

def autolabel_gflops(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('%.3f' % height,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', size=10)

autolabel_gflops(rects_gflops)

plt.ylim(0, max(gflops) * 1.15)
fig.tight_layout()

gflops_out = os.path.join(output_dir, 'gflops_diagram.png')
plt.savefig(gflops_out, dpi=300, bbox_inches='tight')
plt.close()

print(f"Diagrams saved to {output_dir}")
