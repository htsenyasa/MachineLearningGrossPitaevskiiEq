import glob
import sys
import os
sys.path.append("../../Src/")

import analyzer as an

files = []
file_ex = "*.inf"

for file in glob.glob(file_ex):
    files.append(os.path.splitext(file)[0])
files.sort()

for file in files:
    plt = an.load_info(file + ".inf")
    plt.display_plot(file)
    
