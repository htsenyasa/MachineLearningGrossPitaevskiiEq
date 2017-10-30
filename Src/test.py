import argparse

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

parser.add_argument('--interaction-param', type=int, default=0, help = 'specify the interaction parameter [Default = 0]')
parser.add_argument('--random-interaction', type=bool, default=False, help = 'is interaction parameter random ? [Default = False]')
parser.add_argument('--energy-file-name', type=str, default="energy.dat", help = 'file name of energy values ? [Default = energy.dat]')
parser.add_argument('--potential-file-name', type=str, default="energy.dat", help = 'file name of energy values ? [Default = potential.dat]')
parser.add_argument('--test-param', type=int, default=[256, 40, 20, 1], nargs='+')


args = parser.parse_args()
inter_param = args.test_param
print(inter_param)
inter_param = args.interaction_param
print(inter_param)
