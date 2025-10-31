import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(4,3))
ax.plot([0,1],[0,1])
ax.set_ylabel("Y label")
#fig.canvas.draw()                # MUST draw so label positions are computed

# get bbox of the 1st y-tick label (display / pixel coords)
bbox_disp = ax.get_yticklabels()[3].get_window_extent()

# convert to figure coordinates (0..1)
bbox_fig = fig.transFigure.inverted().transform_bbox(bbox_disp)

#%%
# create rectangle in figure coords and add to figure
rect = patches.Rectangle(
    (bbox_fig.x0, bbox_fig.y0), bbox_fig.width, bbox_fig.height,
    transform=fig.transFigure, fill=False, edgecolor="red", linewidth=2
)
fig.add_artist(rect)

plt.show()