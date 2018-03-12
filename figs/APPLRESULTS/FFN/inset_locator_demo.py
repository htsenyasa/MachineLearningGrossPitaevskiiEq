import glob
import sys
import os
#import matplotlib
#matplotlib.use("pgf")
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("../../Src/")
from decimal import Decimal
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import analyzer as an

epochs = [i+1 for i in range(60)]

z1 = an.load_info("potential-g-1-epoch-20-.inf")
z2 = an.load_info("potential-g-1-epoch-40-.inf")
z3 = an.load_info("potential-g-1-epoch-60-.inf")

datum = [z1, z2, z3]
err = []
rel_err = []

err = [x.testdataset - x.predicted for x in datum]
rel_err = [((x.testdataset - x.predicted) / (x.testdataset)) * 100 for x in datum]
axs = []
fig = plt.figure()

for i, data in enumerate(datum):
    ax = fig.add_subplot(1, 4, i + 1)
    axs.append(ax)
    if i != 0: ax = fig.add_subplot(1, 4, i + 1)
    ax.plot(data.testdataset, data.testdataset, "--r", label=None, linewidth = 1, alpha=0.5)
    ax.plot(data.testdataset, data.predicted, ".", label = None, markersize = 2)

    ax.set_xlabel("True Energy", fontsize = 20)
    if i == 0:
        ax.set_ylabel("Predicted Energy", fontsize = 20)

    ax.tick_params(labelsize = 18)
    if i != 0: ax.yaxis.set_ticklabels([])
    ax.legend()
    ax.grid()
    ax.axes.set_aspect(aspect = "equal", adjustable = "box")

    left, bottom, width, height = [0.78, 0.20, .2, .2]
    err_inset = inset_axes(ax, width=1.3, height=1.3, bbox_to_anchor = [left, bottom, width, height], bbox_transform = ax.transAxes)
    err_inset.hist(err[i], range=[-np.amax(np.abs(err[i])), np.amax(np.abs(err[i]))], bins=20)
    #err_inset.set_title("Error", fontsize = 14)
    err_inset.tick_params(labelsize=11)
    err_inset.xaxis.tick_top()



    left2, bottom2, width2, height2 = [0.10, 0.70, .3, .3]
    r_err_inset = inset_axes(ax, width=1.3, height=1.3, bbox_to_anchor = [left2, bottom2, width2, height2], bbox_transform = ax.transAxes)
    r_err_inset.hist(rel_err[i], range=[-np.amax(np.abs(rel_err[i])), np.amax(np.abs(rel_err[i]))], bins=20)
    #r_err_inset.set_title("Rel. Error (%)", fontsize = 14)
    r_err_inset.tick_params(labelsize=11)
    r_err_inset.yaxis.tick_right()


ax4 = fig.add_subplot(144, aspect = "equal", adjustable = "box-forced")
ax4.plot(epochs, datum[2].loss, "--b", label="Loss")
x1, x0 = ax4.get_xlim()
y1, y0 = ax4.get_ylim()
ax4.legend()
ax4.grid()
#plt.gcf()
#plt.yscale("log")
ax4.axes.set_aspect(np.abs(x1-x0) / np.abs(y1-y0), adjustable = "box-forced")
#ax4.set_aspect(((np.log10(x_max / x_min)) / (np.log10(y_max / y_min))))
#ax4.axes.set_aspect("equal", adjustable = "box-forced")
ax4.yaxis.tick_right()
ax4.tick_params(labelsize = 18)

fig.subplots_adjust(wspace = 0.02)




#plt.tight_layout()
#plt.subplot_tool()
plt.show()
#plt.savefig(test + ".svg", format = "svg", dpi=1200)
