import matplotlib.pyplot as plt
import numpy as np

color_map = {"hash": 'orange', "heft": 'green', "jit": 'red', \
              "navigator": 'blue'}
x = [20,40,60,80,100,120,140,160,180,200,220,240]
scheduler_types = ["navigator", "hash"]
nav_used_workers = [20,40,48,46,51,46,50,46,45,46,49,48]
nav_med_slow_down = [74.15,5.65,4.27,4.13,4.462,3.99,3.38,3.95,3.96,3.99,4.53,4.08]
hash_used_workers = [20,40,60,80,100,120,140,160,180,200,220,240]
hash_med_slow_down = [113.74,37.62,17.21,9.49,6.89,4.72,4.18,3.77,3.4,3.19,2.98,2.97]

# Create the figure and the first y-axis
fig, ax1 = plt.subplots(figsize=(24,13))

ax1.set_xlabel('Total Number of Workers Available', fontsize=42,labelpad=14)
ax1.set_ylabel('Number of Workers Used', fontsize=42,labelpad=15)
ax1.plot(x, nav_used_workers, label='Navigator number of used workers', color=color_map["navigator"], linewidth=4)
ax1.plot(x, hash_used_workers, label='Hash number of used workers', color=color_map["hash"], linewidth=4)
ax1.tick_params(axis='y')
plt.xticks(fontsize=42)
plt.yticks(fontsize=42)

# Create dashed lines for the second y-axis
ax2 = ax1.twinx()

ax2.set_ylabel('Median slow_down_factor', fontsize=44,labelpad=15)
ax2.plot(x, nav_med_slow_down, linestyle='--', label='Navigator slow down factor', color=color_map["navigator"], linewidth=4)
ax2.plot(x, hash_med_slow_down, linestyle='--', label='Hash slow down factor', color=color_map["hash"], linewidth=4)
ax2.tick_params(axis='y')

# plt.xticks(fontsize=42)
plt.yticks(fontsize=42)

# Create legends for both categories
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines = lines1 + lines2
labels = labels1 + labels2
ax1.legend(lines, labels, fontsize=40,loc='upper center')

# plt.title('Scaling experiment comparison of Navigator and Hash schemes')


plt.savefig("./schedulers_scalability_comparison_trend.pdf")
plt.show()