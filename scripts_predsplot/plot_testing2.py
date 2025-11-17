import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import ConnectionPatch

np.random.seed(43)
scales = [0.22, 0.3, 3, 0.1, 0.5, 0.2, 1.3, 0.3]
loc = [10, 1, 278, 2, 34, 2, 2389, 1.2]
xs = [np.random.normal(l, l*s, 200) for l,s in zip(loc,scales)]
xs[0] = 1000*xs[0]
xs[1] = np.exp(xs[1])
xs[3] = -np.exp(-xs[3])
xs[5] = np.exp(-xs[5])
xs[7] = -np.exp(xs[7])
# Input xs created

#%%

xs = xs[:2]
x_last = xs[-1]

fig = plt.figure(facecolor=(160/255, 83/255, 237/255), layout="constrained")
h_grid = fig.add_gridspec(1, 3)

fig, axes = plt.subplots(3, len(xs), figsize=(12, 5), layout="constrained", facecolor=(160/255, 83/255, 237/255))
fig.set_constrained_layout_pads(w_pad=0.8, h_pad=0.5, wspace=0, hspace=0)

for ax in axes[0,:]:
    ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
for ax in axes[2,:]:
    ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)


for ax, x in zip(axes[1,:-1], xs[:-1]):
    sns.kdeplot(y=x, fill=True, ax=ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xlabel(None)
    ax.tick_params(bottom=False, labelbottom=False)

sns.kdeplot(y=xs[7], fill=True, ax=axes[1,-1], common_norm=False, bw_adjust=1)
axes[1,-1].invert_xaxis()
axes[1,-1].spines[['top','left','bottom', 'right']].set_visible(False)
axes[1,-1].tick_params(left=False, right=False, labelleft=False)
axes[1,-1].set_yticks([])
axes[1,-1].set_xlabel(None)
#axes[-1].margins(x=0.5)
axes[1,-1].tick_params(bottom=False, labelbottom=False)
#axes[0].plot([-0.5, 10], [0, 0], color="r", lw=4, clip_on=False)


ax8_right = axes[1,-1].twinx()
ax8_right.set_ylabel("Value (right axis)")
ax8_right.set_ylim(axes[1,-1].get_ylim())
ax8_right.spines[['top','left','bottom']].set_visible(False)

con = ConnectionPatch(xyA=(0,mean_xs[0]), coordsA=axes[1,0].transData,
                      xyB=(0,mean_xs[-1]), coordsB=ax8_right.transData)

fig.add_artist(con)
plt.show()



