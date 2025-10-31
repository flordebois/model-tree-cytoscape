import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import ConnectionPatch

# fig = plt.figure(figsize=(8, 4), facecolor=(0.5, 0.5, 0.5))#, layout="constrained")
# ax = fig.add_subplot(221)
#
# ax.plot([5, 10000000], [800000, 900])
# ax2 = fig.add_subplot(2,2,4)
# ax2.plot([5, 10], [800000, 900])
#
# bbox = ax.get_position()
# print(bbox)
# new_bbox = (bbox.x0, bbox.y0-0.5, bbox.width, bbox.height)
# ax3 = fig.add_subplot(2,2,2)
# ax3.set_position(new_bbox)
# ax3.plot([5, 10], [8, 10])


fig = plt.figure(facecolor=(160/255, 83/255, 237/255))#, layout="constrained")
#fig.set_constrained_layout_pads(hspace=0.2)
gs0 = fig.add_gridspec(1, 3, wspace=0.6)

ax1 = fig.add_subplot(gs0[0])

# subgrid with custom height ratios
gssub = gs0[1].subgridspec(3, 1, height_ratios=[1, 7, 4],hspace=0.45)
sub_axes = [fig.add_subplot(gssub[i, 0]) for i in range(3)]
sub_axes[0].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
sub_axes[1].plot([1, 3], [3,-330000], 'k')
sub_axes[1].tick_params(bottom=False, labelbottom=False)
sub_axes[2].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

gssub = gs0[2].subgridspec(3, 1, height_ratios=[1, 7, 4],hspace=0.5)
sub_axes = [fig.add_subplot(gssub[i, 0]) for i in range(3)]
sub_axes[0].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
sub_axes[1].plot([1, 3], [1,30], 'k')
sub_axes[1].tick_params(bottom=False, labelbottom=False)
sub_axes[2].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

# ax1 = fig.add_subplot(gssub[1, 0])
# ax1.plot([1, 3], [-0.400, -0.0000000000465], 'k')
# ax1.tick_params(bottom=False, labelbottom=False)

#plt.tight_layout()
plt.show()