import glob
import sys
import os
sys.path.append("../../Src/")

files = []
file_ex = "*.svg"

for file in glob.glob(file_ex):
    files.append(file)

for src in files:
    dst = src.replace("0.", "0_", 1)
    os.rename(src, dst)
