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


fig = plt.figure()
ax1 = fig.add_subplot(141)

ax1.plot(datum[0].testdataset, datum[0].testdataset, "--r", label=None, linewidth = 1, alpha=0.5)
ax1.plot(datum[0].testdataset, datum[0].predicted, ".", label = None, markersize = 2)
ax1.set_xlabel("True Energy", fontsize = 20)
ax1.set_ylabel("Predicted Energy", fontsize = 20)
ax1.tick_params(labelsize = 18)
ax1.legend()
ax1.grid()
ax1.axes.set_aspect(aspect = "equal", adjustable = "box")
inset_axes = inset_axes(ax1, width="50%", height=1.0, loc="1")

plt.hist(err[0])
plt.title("Relative Error")
plt.xticks([])
plt.yticks([])

ax2 = fig.add_subplot(142)
ax2.plot(datum[1].testdataset, datum[1].testdataset, "--r", label=None, linewidth = 1, alpha=0.5)
ax2.plot(datum[1].testdataset, datum[1].predicted, ".", label = None, markersize = 2)
ax2.set_xlabel("True Energy", fontsize = 20)
ax2.tick_params(labelsize = 18)
ax2.legend()
ax2.grid()
ax2.axes.set_aspect(aspect = "equal", adjustable = "box")

#ax3 = fig.add_subplot(143)
#ax3.plot(datum[2].testdataset, datum[2].testdataset, "--r", label=None, linewidth = 1, alpha=0.5)
#ax3.plot(datum[2].testdataset, datum[2].predicted, ".", label = None, markersize = 2)
#ax3.set_xlabel("True Energy", fontsize = 20)
#ax3.tick_params(labelsize = 18)
#ax3.legend()
#ax3.grid()
#ax3.axes.set_aspect(aspect = "equal", adjustable = "box")
#
#ax4 = fig.add_subplot(144, aspect = "equal", adjustable = "box-forced")
#ax4.plot(epochs, datum[2].loss, "--b", label="Loss")
#x0, x1 = ax4.get_xlim()
#y0, y1 = ax4.get_ylim()
#ax4.axes.set_aspect(np.abs(x1-x0) / np.abs(y1-y0), adjustable = "box-forced")
#

#fig.set_size_inches(8,8)
plt.tight_layout()
#plt.subplot_tool()

plt.show()

#fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey = True)
##left, bottom, width, height = [0.65, 0.20, .2, .2]
##inset = fig.add_axes([left, bottom, width, height])
##left2, bottom2, width2, height2 = [0.19, 0.60, .2, .2]
##inset2 = fig.add_axes([left2, bottom2, width2, height2])
##ax1.scatter(self.testdataset, self.predicted, c = np.abs(err), s = 4)
##ax1.set_title("FNN{}".format(self.arch))
#
#ax1.plot(datum[0].testdataset, datum[0].testdataset, "--r", label=None, linewidth = 1, alpha=0.5)
#ax1.plot(datum[0].testdataset, datum[0].predicted, ".", label = None, markersize = 2)
#ax1.set_xlabel("True Energy", fontsize = 20)
#ax1.set_ylabel("Predicted Energy", fontsize = 20)
#ax1.tick_params(labelsize = 18)
#ax1.legend()
#ax1.grid()
#ax1.axes.set_aspect("equal", "box-forced")
#
#
#ax2.plot(datum[1].testdataset, datum[1].testdataset, "--r", label=None, linewidth = 1, alpha=0.5)
#ax2.plot(datum[1].testdataset, datum[1].predicted, ".", label = None, markersize = 2)
#ax2.set_xlabel("True Energy", fontsize = 20)
#ax2.tick_params(labelsize = 18)
#ax2.legend()
#ax2.grid()
#ax2.axes.set_aspect("equal", "box-forced")
#
#
#ax3.plot(datum[2].testdataset, datum[2].testdataset, "--r", label=None, linewidth = 1, alpha=0.5)
#ax3.plot(datum[2].testdataset, datum[2].predicted, ".", label = None, markersize = 2)
#ax3.set_xlabel("True Energy", fontsize = 20)
#ax3.tick_params(labelsize = 18)
#ax3.legend()
#ax3.grid()
#ax3.axes.set_aspect("equal", "box-forced")
#
#
#    #props = dict(boxstyle='square', facecolor='white', alpha=0.5)
#    #textstr = "MSE:{:.4E}\nTraining Length:{}\nEpoch:{}\nBatch:{}".format(Decimal(float(self.error)), self.training_len, self.cur_epoch, self.batch_size)
#    #ax1.text(0.03, 0.85, textstr, transform=ax1.transAxes, fontsize=11, verticalalignment='top', bbox=props)
#    #ax1.text(, , "{}\nlr={}\nepoch={}\ntrain_len={}\ntest_len={}\nerror={}".format(self.arch, self.learning_rate, self.cur_epoch, self.training_len, self.test_len, self.error))
#
#    #inset.hist(err, range=[-np.amax(np.abs(err)), np.amax(np.abs(err))], bins=20)
#    #inset2.hist(relative_err, range=[-np.amax(np.abs(relative_err)), np.amax(np.abs(relative_err))], bins=20)
#    #inset.set_title("Error", fontsize = 18)
#    #inset.tick_params(labelsize=12)
#    #inset2.set_title("Rel.Error (%)", fontsize = 18)
#    #inset2.tick_params(labelsize=12)
#
#figure = plt.gcf()
#figure.set_size_inches(8,8)
#plt.tight_layout()
##plt.show()
##plt.savefig("test" + ".svg", format = "svg", dpi=1200)
#
