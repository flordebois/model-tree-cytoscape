import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from matplotlib.patches import ConnectionPatch

TICK_WIDTH_ABS = 0.65
LABEL_WIDTH_ABS = 0.25
OUTER_VERTICAL_MARGIN_ABS = 0.2
INNER_VERTICAL_MARGIN_ABS = 0.2
EXTRA_WIDTH_ABS = 0.15

def predsplot(X, coefficients, y_hat, n_max=5, fig_size=(12,8), intercept = None, displaytype = "histogram",
              truncate_total_pred = False, variable_tick_width = True, file_directory=None, highlight_x=None):
    if intercept is None:
        center_pred = np.mean(y_hat)
        pred = y_hat - center_pred
        contributions = coefficients * (X - np.mean(X, axis=0))
        center_name = "centercept"
    else:
        center_pred = intercept
        pred = y_hat - intercept
        contributions = coefficients * X
        center_name = "intercept"


    contributions_pred_max = np.max(contributions, axis=0)
    contributions_pred_min = np.min(contributions, axis=0)

    contributions_var = np.var(contributions, axis=0)
    sorted_X = np.argsort(contributions_var)[::-1]

    n_coefficients = np.sum(np.abs(coefficients) > 10**-14) # only select variables where coefficient is not zero
    n = np.minimum(n_coefficients, n_max)
    fig_width, fig_height = fig_size
    if fig_height < 3:
        warnings.warn("Plot height is small, the default titles will be cropped. Consider making the figure larger or higher.")

    outer_vertical_margin = OUTER_VERTICAL_MARGIN_ABS / fig_height
    inner_vertical_margin = INNER_VERTICAL_MARGIN_ABS / fig_height
    tick_width = TICK_WIDTH_ABS / fig_width
    label_width = LABEL_WIDTH_ABS / fig_width

    extra_width = EXTRA_WIDTH_ABS / fig_width

    width_left_over = 1 - (n + 2) * tick_width - 2 * extra_width - 2 * label_width
    plot_width = width_left_over / (n + 1)
    if plot_width < 0:
        if variable_tick_width:
            plot_width = 0.5/fig_width #Not important gets adjusted later and new_plot_width is used
        else:
            raise Exception(f"Error: plot is too narrow, inner plot width is {plot_width}. Consider making the figure larger or wider.")
    elif not variable_tick_width and plot_width < tick_width:
        warnings.warn("Plot width is small. Consider making the figure larger or wider.")

    left_margin = label_width + tick_width + extra_width

    fig = plt.figure(figsize=(fig_width, fig_height))

    outer_plot_x0 = label_width + tick_width
    outer_plot_y0 = outer_vertical_margin
    outer_plot_width = 1 - 2 * (tick_width + label_width)
    outer_plot_height = 1 - 2 * outer_vertical_margin

    max_pred = np.max(contributions_pred_max) if truncate_total_pred else np.maximum(np.max(pred), np.max(contributions_pred_max))
    min_pred = np.min(contributions_pred_min) if truncate_total_pred else np.minimum(np.min(pred), np.min(contributions_pred_min))
    inner_vertical_margin_pred = inner_vertical_margin * (max_pred - min_pred) / (
                outer_plot_height - 2 * inner_vertical_margin)

    outer_plot_max = max_pred + inner_vertical_margin_pred
    outer_plot_min = min_pred - inner_vertical_margin_pred

    ax_outer = fig.add_axes([outer_plot_x0, outer_plot_y0, outer_plot_width, outer_plot_height])
    ax_outer.invert_xaxis()
    if displaytype == "density":
        prediction_display = sns.kdeplot(y=pred, fill=True, ax=ax_outer, color="C0")
    elif displaytype == "histogram":
        prediction_display = sns.histplot(y=pred, fill=True, ax=ax_outer, color="C0")


    ax_outer.margins(
        x=((2 * extra_width + n * (tick_width + plot_width)) / plot_width))  # margin is ... times range of y-values
    ax_outer.spines[['top', 'bottom', 'right']].set_visible(False)
    ax_outer.tick_params(bottom=False, labelbottom=False)
    ax_outer.ticklabel_format(scilimits=[-3, 4])
    ax_outer.set_ylim([outer_plot_min, outer_plot_max])
    ax_outer.set_xlabel(None)
    ax_outer.set_ylabel(f'Prediction contribution{'' if intercept is not None else " (relative to mean)"}')

    ax_right = ax_outer.twinx()
    ax_right.ticklabel_format(scilimits=[-3, 4])
    ax_right.spines[['top', 'left', 'bottom']].set_visible(False)
    ax_right.set_ylim([outer_plot_min + center_pred, outer_plot_max + center_pred])
    ax_right.set_ylabel(f"Predictions (with {center_name} = {np.round(center_pred, 2)})")

    sub_axes = [None] * n
    sub_axes_tick_width = np.empty(n)
    y0s = np.empty(n)
    plot_heights = np.empty(n)
    for i in range(n):
        idx_x = sorted_X[i]
        coef = coefficients[idx_x]
        x = X[:, idx_x]

        inner_plot_max = contributions_pred_max[idx_x]
        inner_plot_min = contributions_pred_min[idx_x]

        y0s[i] = (inner_plot_min - outer_plot_min) * outer_plot_height/(outer_plot_max - outer_plot_min) + outer_vertical_margin

        plot_heights[i] = (inner_plot_max - inner_plot_min) * outer_plot_height/(outer_plot_max - outer_plot_min)

        x0 = left_margin + tick_width + i * (tick_width + plot_width)
        sub_axes[i] = fig.add_axes([x0, y0s[i], plot_width, plot_heights[i]]) # x0, y0, width, height
        if coef > 0:
            plot_color = "green"
            sub_axes[i].set_ylim([np.min(x), np.max(x)])
        else:
            plot_color = "firebrick"
            sub_axes[i].set_ylim([np.max(x), np.min(x)])
        if displaytype == "density":
            sns.kdeplot(y=x, fill=True, ax=sub_axes[i], bw_adjust=1, color=plot_color)
        elif displaytype == "histogram":
            sns.histplot(y=x, fill=True, ax=sub_axes[i], color=plot_color)
        sub_axes[i].spines[['top', 'right', 'bottom']].set_visible(False)
        sub_axes[i].set_xlabel(f"X{idx_x}")
        sub_axes[i].tick_params(bottom=False, labelbottom=False)
        sub_axes[i].ticklabel_format(scilimits=[-3, 4])

        sub_ax_ticks_labels = sub_axes[i].get_yticklabels()[1:-1]
        if len(sub_ax_ticks_labels) < 1:
            sub_ax_ticks_labels = sub_axes[i].get_yticklabels()
        sub_ax_ticks_boxes = [fig.transFigure.inverted().transform_bbox(tick_label.get_window_extent()) for tick_label in sub_ax_ticks_labels]
        sub_axes_tick_width[i] = max([x0 - ticks_box.x0 for ticks_box in sub_ax_ticks_boxes])

    if variable_tick_width:
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

        if new_plot_width < 0:
            raise Exception(
                f"Error: plot is too narrow, inner plot width is {plot_width}. Consider making the figure larger or wider.")

        for i in range(n):
            new_x0 = new_left_margin + np.sum(sub_axes_tick_width[:i+1]) + i*new_plot_width
            sub_axes[i].set_position([new_x0, y0s[i], new_plot_width, plot_heights[i]])

        ax_outer.margins(x=(2 * extra_width + np.sum(sub_axes_tick_width) + n*new_plot_width) / new_plot_width)

        fig.add_artist(ConnectionPatch(xyA=(0, 0), coordsA=ax_outer.transData,
                                   xyB=(prediction_display.dataLim.xmax * (
                                           (2 * extra_width + np.sum(sub_axes_tick_width) + n*new_plot_width) / new_plot_width + 1), 0)
                                   , coordsB=ax_outer.transData))

    else:
        fig.add_artist(ConnectionPatch(xyA=(0, 0), coordsA=ax_outer.transData,
                                   xyB=(prediction_display.dataLim.xmax * (
                                       (2 * extra_width + n * (tick_width + plot_width)) / plot_width + 1), 0)
                                   , coordsB=ax_outer.transData))



    if file_directory is not None:
        plt.close()
        fig.savefig(file_directory)
    else:
        plt.show()

    return None