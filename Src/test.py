import argparse

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

parser.add_argument('--interaction-param', type=int, default=0, help = 'specify the interaction parameter [Default = 0]')
parser.add_argument('--random-interaction', type=bool, default=False, help = 'is interaction parameter random ? [Default = False]')
parser.add_argument('--energy-file-name', type=str, default="energy.dat", help = 'file name of energy values ? [Default = energy.dat]')
#parser.add_argument('--potential-file-name', type=str, default="energy.dat", help = 'file name of energy values ? [Default = potential.dat]')
parser.add_argument('--test-param', type=int, default=[256, 40, 20, 1], nargs='+')
parser.add_argument('--data-file',            type=str,             default="potential.dat",                   help = 'file name of energy values ? [Default = energy.dat]')
#parser.add_argument('--potential-file-name',  type=str,             default="energy.dat",                      help = 'file name of energy values ? [Default = potential.dat]')
parser.add_argument('--label-file',         type=str,             default="energy.dat",                      help = 'file name of energy values ? [Default = energy.dat]')
parser.add_argument('--network-arch',       type=int,             default=[256, 40, 20, 1],      nargs='+',      help = 'Network arch : (default: 256-40-20-1)')

args = parser.parse_args()
inter_param = args.network_arch
print(inter_param)
inter_param = args.label_filename
print(inter_param)
