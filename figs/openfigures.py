import glob
import sys
sys.path.append("../Src/")

import analyzer as an

files = []
for file in glob.glob("*.inf"):
    files.append(file)

files.sort()



i = 0
#print(">>>")
command = input(">>>")
while command != 'exit':
    if i == len(files):
        print("end")
    if command == 'next' :
        i += 1
    elif command == 'prev':
        i -= 1
    else: pass

    plot = an.load_info(files[i])
    plot.display_plot()
    command = input()
