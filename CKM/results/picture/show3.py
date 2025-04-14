import pandas as pd
import matplotlib.pyplot as plt

# 重新加载数据并整理成DataFrame
data_bar_chart = {
    'Latent': [50, 63, 195, 292, 305, 372, 380, 405, 452, 489, 495, 527, 746, 756, 798, 800, 912, 1018, 1053, 1126, 1132, 1137, 1147, 1179, 1192, 1212, 1321, 1390, 1436, 1477, 1497, 1549, 1677, 1734, 1736, 1738, 1763, 1820, 1859, 1877, 1913, 1926, 1957],
    'GCG': [0.687, 0.709, 0.578, 0.776, 0.657, 0.601, 0.903, 0.985, 0.556, 0.772, 0.590, 0.575, 0.675, 0.586, 0.608, 0.500, 0.642, 0.690, 0.560, 0.631, 0.612, 0.750, 0.716, 0.836, 0.675, 0.690, 0.832, 0.601, 0.757, 0.571, 0.698, 0.560, 0.653, 0.795, 0.690, 0.534, 0.757, 0.817, 0.821, 0.604, 0.881, 0.761, 0.634],
    'AmpleGCG': [0.620, 0.640, 0.660, 0.680, 0.530, 0.550, 0.870, 0.970, 0.630, 0.560, 0.560, 0.720, 0.590, 0.500, 0.680, 0.630, 0.560, 0.630, 0.560, 0.570, 0.570, 0.560, 0.660,  0.860, 0.600, 0.590, 0.760, 0.500, 0.620, 0.600, 0.570, 0.530, 0.700, 0.650, 0.580, 0.590, 0.680, 0.530, 0.550, 0.620, 0.630, 0.600, 0.580],
    'AmpleGCG-plus': [0.620, 0.550, 0.610, 0.620, 0.560,0.570, 0.880, 0.940, 0.600, 0.730, 0.550, 0.760, 0.590, 0.510, 0.630, 0.610, 0.630, 0.710, 0.630, 0.540, 0.610, 0.570, 0.620, 0.870, 0.640, 0.560, 0.670, 0.500, 0.670, 0.610, 0.610, 0.570, 0.700, 0.570, 0.590, 0.650, 0.630, 0.520, 0.630, 0.550, 0.570, 0.640, 0.530]
}

# 创建DataFrame
df_bar_chart = pd.DataFrame(data_bar_chart)

# 按照Latent列排序
df_bar_chart = df_bar_chart.sort_values(by='Latent')

# 绘制柱状图
fig, ax = plt.subplots(figsize=(12, 8))

# 设置每个Latent值的X位置
x = range(len(df_bar_chart))

# 定义宽度和位置
width = 0.25  # 柱子的宽度
# 创建柱状图
ax.bar(x, df_bar_chart['GCG'], width, label='GCG', color='tab:blue')
ax.bar([p + width for p in x], df_bar_chart['AmpleGCG'], width, label='AmpleGCG', color='tab:orange')
ax.bar([p + width*2 for p in x], df_bar_chart['AmpleGCG-plus'], width, label='AmpleGCG-plus', color='tab:green')

# 添加标签和标题
ax.set_xlabel('Latent')
ax.set_ylabel('Scores')
ax.set_title('Comparison of GCG, AmpleGCG, and AmpleGCG-plus across Latents')

# 设置X轴刻度
ax.set_xticks([p + width for p in x])
ax.set_xticklabels(df_bar_chart['Latent'])

# 设置纵轴范围和区间，确保从0.5开始
ax.set_ylim(0.4, 1)  # 纵轴从0.5到1.05
ax.yaxis.set_ticks([i * 0.1 for i in range(4, 11)])  # 设置每0.05一个刻度，确保从0.5开始

# 添加纵轴的虚线
for tick in ax.get_yticks():
    ax.axhline(tick, color='gray', linestyle='--', alpha=0.5, zorder=0)

# 添加图例
ax.legend()

plt.xticks(rotation=90)  # X轴标签旋转90度，防止重叠
plt.tight_layout()
plt.show()