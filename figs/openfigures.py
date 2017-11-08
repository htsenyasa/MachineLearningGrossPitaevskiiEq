import glob
import sys
sys.path.append("../Src/")

import analyzer as an

files = []




command = input(">>> FFN or CNN :")
if command == 'FFN':
    file_path = "./FFN/*.inf"
elif command == 'CNN':
    file_path = "./CNN/*.inf"
else:
    print("Wrong argument!")
    sys.exit(0)

for file in glob.glob(file_path):
    files.append(file)
files.sort()



i = 0
command = input(">>>")
while command != 'exit':
    if i == len(files) - 1:
        print("end")
        sys.exit(0)
    if command == 'n' :
        i += 1
    elif command == 'p':
        i -= 1
    else: pass

    plot = an.load_info(files[i])
    plot.display_plot()
    command = input()
