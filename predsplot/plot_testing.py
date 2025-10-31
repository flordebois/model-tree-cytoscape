import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

scales = [1, 0.5, 2.2, 1.8, 0.2, 0.8, 1.3]
xs = [np.random.normal(loc=0, scale=scales[i], size=200) for i in range(7)]
xgrid = np.linspace(-5, 10, 200)

fig, axes = plt.subplots(len(xs), 1, sharex=True, figsize=(4, 8))

for ax, x in zip(axes, xs):
    kde = gaussian_kde(x)
    y = kde(xgrid)
    y /= y.max()
    ax.fill_betweenx(y, xgrid, alpha=0.5)
    ax.axhline(0, color='k', lw=1)
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    ax.tick_params(left=True, labelleft=True, bottom=False, labelbottom=False)

axes[-1].set_xlabel("Density (normalized)")
axes[len(xs)//2].set_ylabel("Value")
plt.tight_layout()
plt.show()

# Vertical version 2
# fig, ax = plt.subplots(figsize=(8, 6))
# max_height = 0.7  # desired (identical) horizontal width for each density
# n = len(xs)
#
# # plotting densities (horizontal orientation, shifted by index)
# for i, x in enumerate(xs):
#     kde = gaussian_kde(x)
#     y = kde(xgrid)
#     y = y / y.max() * max_height            # normalize heights to same max
#     ax.fill_betweenx(xgrid, i, i + y, alpha=0.5, linewidth=0)
#
# # draw a horizontal zero line visible across all densities
# ax.axhline(0, color='k', lw=1)
#
# # add a small vertical "axis" (with arrow) at the flat side of each density
# ymin, ymax = xgrid.min(), xgrid.max()
# tick_vals = np.linspace(ymin, ymax, 5)      # ticks for every density (same scale)
# tick_len = 0.06                             # length of tick marks (in x-units)
# label_offset = 0.22                         # spacing between tick and label
#
# for i in range(n):
#     x_pos = i - 0.08                        # place axis slightly left of flat side
#     # vertical line with arrow pointing upward
#     ax.plot([x_pos+0.075, x_pos+0.075], [ymin, ymax], color='k', lw=0.8)
#     # ticks + labels (same numeric scale as main y axis)
#     for tv in tick_vals:
#         ax.plot([x_pos - tick_len, x_pos], [tv, tv], color='k', lw=0.9)
#         ax.text(x_pos - tick_len - label_offset, tv, f"{tv:.1f}",
#                 va='center', ha='left', fontsize=7)
#
# # cosmetic limits & labels
# ax.set_xlim(-0.6, n - 1 + max_height + 0.2)
# ax.set_ylim(ymin - 0.5, ymax + 0.5)
# ax.set_xlabel('density index + horizontal shift')
# ax.set_ylabel('value (shared vertical scale)')
# plt.tight_layout()
# plt.show()


# Vertical version 1
# max_height = 0.8
# for i, x in enumerate(xs):
#     kde = gaussian_kde(x)
#     y = kde(xgrid)
#     y /= y.max() / max_height  # normalize heights
#     plt.fill_betweenx(xgrid, i, y + i, alpha=0.5)
#     plt.text(i + 0.4, xgrid[0] - 1, f"{i}", ha='center')  # vertical axis labels
#
# plt.axhline(0, color='k', lw=1)
# plt.show()