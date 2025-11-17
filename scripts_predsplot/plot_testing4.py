import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from matplotlib.patches import ConnectionPatch
from numpy.f2py.auxfuncs import throw_error
import matplotlib.patches as patches

np.random.seed(43)
coefficients = np.array([-1 / 1000, 10, 1, -100, 1 / 1000000, 40, 1 / 2000])
scales = [0.22, 0.3, 2, 0.1, 0.5, 0.2, 1.3]
loc = [10, 1, 1, 2, 34, 2, 2389]
xs = [np.random.normal(l, l*s, 200) for l,s in zip(loc,scales)]
xs[0] = 1000*xs[0]
#xs[1] = np.exp(xs[1])
xs[3] = -np.exp(-xs[3])
xs[5] = np.exp(-xs[5])
#xs[1][-1] = max(xs[1])+1.6
xs[4] = xs[4]*100000

n_data = 4
xs = np.column_stack(xs[:n_data])
coefficients = coefficients[:n_data]
pred_with_center = np.random.normal(1.2, 1.2 * 0.3, 200)
pred_with_center = -np.exp(pred_with_center)*5

#%%
center_pred = np.mean(pred_with_center)
pred = pred_with_center - center_pred

contributions = coefficients * (xs - np.mean(xs, axis=0))
contributions_pred_max = np.max(contributions, axis=0)
contributions_pred_min = np.min(contributions, axis=0)

contributions_var = np.var(contributions, axis=0)
sorted_xs = np.argsort(contributions_var)[::-1]

n = xs.shape[1]
fig_width = 8
fig_height = 4
tick_width_abs = 0.75
tick_width = tick_width_abs / fig_width
label_width_abs = 0.25
label_width = label_width_abs / fig_width
outer_vertical_margin_abs = 0.2
outer_vertical_margin = outer_vertical_margin_abs / fig_height
inner_vertical_margin_abs = 0.2
inner_vertical_margin = inner_vertical_margin_abs / fig_height
extra_width_abs = 0.15
extra_width = extra_width_abs / fig_width

width_left_over = 1 - (n+2)*tick_width - 2*extra_width - 2*label_width
plot_width = width_left_over/(n+1)
if plot_width < 0.05/fig_height:
    raise Exception(f"Error: plot is too narrow, inner plot width is {plot_width}. Consider making the figure larger or wider.")
elif plot_width < tick_width:
    warnings.warn("Plot width is small. Consider making the figure larger or wider.")

left_margin = label_width + tick_width + extra_width

fig = plt.figure(figsize=(fig_width, fig_height))#, facecolor=(160/255, 83/255, 237/255))
#fig, ax = plt.subplots(figsize=(12, 3), facecolor=(160/255, 83/255, 237/255))

outer_plot_x0 = label_width + tick_width
outer_plot_y0 = outer_vertical_margin
outer_plot_width = 1 - 2 * (tick_width + label_width)
outer_plot_height = 1 - 2 * outer_vertical_margin

max_pred = np.maximum(np.max(pred), np.max(contributions_pred_max))
min_pred = np.minimum(np.min(pred), np.min(contributions_pred_min))
inner_vertical_margin_pred = inner_vertical_margin*(max_pred - min_pred)/(outer_plot_height - 2*inner_vertical_margin)

outer_plot_max = max_pred + inner_vertical_margin_pred
outer_plot_min = min_pred - inner_vertical_margin_pred

ax_outer = fig.add_axes([outer_plot_x0, outer_plot_y0, outer_plot_width, outer_plot_height])
ax_outer.invert_xaxis()
prediction_density = sns.kdeplot(y=pred, fill=True, ax=ax_outer, bw_adjust=1) # bw_adjust -> smoothness

ax_outer.margins(x=((2 * extra_width + n * (tick_width + plot_width)) / plot_width)) # margin is ... times range of y-values
ax_outer.spines[['top', 'bottom', 'right']].set_visible(False)
#ax_outer.tick_params(bottom=False, labelbottom=False)
ax_outer.set_ylim([outer_plot_min, outer_plot_max])
ax_outer.set_xlabel(None)
ax_outer.set_ylabel('Prediction contribution (relative to mean prediction)')

ax_right = ax_outer.twinx()
ax_right.spines[['top','left','bottom']].set_visible(False)
ax_right.set_ylim([outer_plot_min + center_pred, outer_plot_max + center_pred])
ax_right.set_ylabel(f"Predictions (with centercept = {np.round(center_pred,2)})")


sub_axes = [None] * n
sub_axes_tick_width = np.empty(n)
y0s = np.empty(n)
plot_heights = np.empty(n)
for i in range(n):
    idx_x = sorted_xs[i]
    coef = coefficients[idx_x]
    x = xs[:, idx_x]

    inner_plot_max = contributions_pred_max[idx_x]
    inner_plot_min = contributions_pred_min[idx_x]

    y0s[i] = (inner_plot_min - outer_plot_min) * outer_plot_height/(outer_plot_max - outer_plot_min) + outer_vertical_margin

    plot_heights[i] = (inner_plot_max - inner_plot_min) * outer_plot_height/(outer_plot_max - outer_plot_min)

    x0 = left_margin + tick_width + i * (tick_width + plot_width)
    sub_axes[i] = fig.add_axes([x0, y0s[i], plot_width, plot_heights[i]]) # x0, y0, width, height
    if coef > 0:
        sns.kdeplot(y=x, fill=True, ax=sub_axes[i], bw_adjust=1, color="green")
        sub_axes[i].set_ylim([np.min(x), np.max(x)])
    else:
        sns.kdeplot(y=x, fill=True, ax=sub_axes[i], bw_adjust=1, color="red")
        sub_axes[i].set_ylim([np.max(x), np.min(x)])
    sub_axes[i].spines[['top', 'right', 'bottom']].set_visible(False)
    sub_axes[i].set_xlabel(None)
    sub_axes[i].tick_params(bottom=False, labelbottom=False)

    sub_ax_ticks_labels = sub_axes[i].get_yticklabels()[1:-1]
    sub_ax_ticks_boxes = [fig.transFigure.inverted().transform_bbox(tick_label.get_window_extent()) for tick_label in sub_ax_ticks_labels]
    sub_axes_tick_width[i] = max([x0 - ticks_box.x0 for ticks_box in sub_ax_ticks_boxes])



outer_ax_ticks_boxes = [fig.transFigure.inverted().transform_bbox(tick_label.get_window_extent()) for tick_label in ax_outer.get_yticklabels()]
outer_ax_tick_width = max([outer_plot_x0 - ticks_box.x0 for ticks_box in outer_ax_ticks_boxes])
ax_right_ticks_boxes = [fig.transFigure.inverted().transform_bbox(tick_label.get_window_extent()) for tick_label in ax_right.get_yticklabels()]
ax_right_tick_width = max([ticks_box.x1 - (outer_plot_x0 + outer_plot_width) for ticks_box in ax_right_ticks_boxes])

new_outer_plot_x0 = label_width + outer_ax_tick_width
new_outer_plot_width = 1 - 2*label_width - outer_ax_tick_width - ax_right_tick_width
ax_outer.set_position([new_outer_plot_x0, outer_plot_y0, new_outer_plot_width, outer_plot_height])

new_width_left_over = 1 - np.sum(sub_axes_tick_width) - outer_ax_tick_width - ax_right_tick_width - 2*extra_width - 2*label_width
new_plot_width = new_width_left_over/(n+1)
new_left_margin = label_width + outer_ax_tick_width + extra_width

# rect = patches.Rectangle(
#     (new_left_margin, 0), np.sum(sub_axes_tick_width[:1+1]) + new_plot_width, 1,
#     transform=fig.transFigure, fill=False, edgecolor="red", linewidth=1
# )
# fig.add_artist(rect)
for i in range(n):
    new_x0 = new_left_margin + np.sum(sub_axes_tick_width[:i+1]) + i*new_plot_width
    sub_axes[i].set_position([new_x0, y0s[i], new_plot_width, plot_heights[i]])
    # rect = patches.Rectangle(
    #     (new_x0, 0), new_plot_width, 1,
    #     transform=fig.transFigure, fill=False, edgecolor="orange", linewidth=0.4
    # )
    # fig.add_artist(rect)

ax_outer.margins(x=(2 * extra_width + np.sum(sub_axes_tick_width) + n*new_plot_width) / new_plot_width)

fig.add_artist(ConnectionPatch(xyA=(0, 0), coordsA=ax_outer.transData,
                               xyB=(prediction_density.dataLim.xmax * (
                                       (2 * extra_width + np.sum(
                                           sub_axes_tick_width) + n * new_plot_width) / new_plot_width + 1), 0)
                               , coordsB=ax_outer.transData))

# fig.add_artist(ConnectionPatch(xyA=(0,0), coordsA=ax_outer.transData,
#                       xyB=(prediction_density.dataLim.xmax*((2 * extra_width + n * (tick_width + plot_width)) / plot_width + 1), 0)
#                       , coordsB=ax_outer.transData))



plt.show()
