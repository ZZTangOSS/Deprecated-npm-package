import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
import numpy as np

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']

base_size = 15 
mpl.rcParams['font.size'] = base_size
mpl.rcParams['axes.labelsize'] = base_size + 1
mpl.rcParams['xtick.labelsize'] = base_size
mpl.rcParams['ytick.labelsize'] = base_size
mpl.rcParams['axes.linewidth'] = 0.6

categories = ['Critical', 'High', 'Moderate', 'Low', 'All Vulnerabilities']

sum_non_gdnp = np.array([71001481, 154985103, 254204780, 5766784])
sum_gdnp = np.array([121947, 124555264, 22005997, 5172949])

pkg_counts_non_gdnp = np.array([12, 52, 33, 4])
pkg_counts_gdnp = np.array([1, 6, 9, 2])

avg_non_gdnp = sum_non_gdnp / pkg_counts_non_gdnp
avg_gdnp = sum_gdnp / pkg_counts_gdnp

total_pkg_non_gdnp = 84
total_pkg_gdnp = 13

total_val_non_gdnp = np.sum(sum_non_gdnp) / total_pkg_non_gdnp
total_val_gdnp = np.sum(sum_gdnp) / total_pkg_gdnp

data_left_plot = np.append(avg_non_gdnp, total_val_non_gdnp)
data_right_plot = np.append(avg_gdnp, total_val_gdnp)

global_max = max(np.max(data_left_plot), np.max(data_right_plot))

fig, ax = plt.subplots(figsize=(7, 3.5))

x = np.arange(len(categories)) 
total_width = 0.7              
bar_width = total_width / 2    

color_left = '#2D85C1' 
color_right = '#C0382A'

bars1 = ax.bar(x - bar_width/2, data_left_plot, width=bar_width, 
               color=color_left, edgecolor='black', linewidth=0.6, 
               zorder=3, label='Non-GDNP')

bars2 = ax.bar(x + bar_width/2, data_right_plot, width=bar_width, 
               color=color_right, edgecolor='black', linewidth=0.6, 
               zorder=3, label='GDNP')

def millions_formatter(x_val, pos):
    if x_val == 0: return '0'
    return f'{x_val*1e-6:.0f}M'

ax.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
ax.set_ylim(0, global_max * 1.25) 

ax.set_ylabel('Average Risk Exposure Count', labelpad=5)
ax.set_xticks(x)
ax.set_xticklabels(categories)

ax.tick_params(axis='x', which='both', length=0) 
ax.tick_params(axis='y', which='both', length=0) 

ax.set_axisbelow(True)
ax.grid(axis='y', linestyle='--', alpha=0.4, color='#999999', zorder=0)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        value_in_millions = height / 1e6
        
        offset = 3
        
        ax.annotate(f'{value_in_millions:.1f}M',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, offset), 
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold', color='black')

autolabel(bars1)
autolabel(bars2)

ax.legend(loc='upper right', frameon=False, fontsize=12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')

plt.tight_layout()
output_filename = 'fig_avg_risk_exposure_with_total.pdf'
plt.savefig(output_filename, format='pdf', bbox_inches='tight', dpi=600)
print(f"Output file generated: {output_filename}")