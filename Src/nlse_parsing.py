import argparse

def process_parsing(arch, desc = "NLSE arguments"):
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--batch-size',        type=int,             default=arch["batch_size"],                 metavar='N',    help = 'input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size',   type=int,             default=arch["test_batch_size"],               metavar='N',    help = 'input batch size for testing (default: 1000)')
    parser.add_argument('--epochs',            type=int,             default=arch["epoch"],                 metavar='N',    help = 'number of epochs to train (default: 10)')
    parser.add_argument('--lr',                type=float,           default=arch["lr"],               metavar='LR',   help = 'learning rate (default: 0.01)')
    parser.add_argument('--momentum',          type=float,           default=0.2,                metavar='M',    help = 'SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda',           action='store_true',  default=False,                              help = 'disables CUDA training')
    parser.add_argument('--seed',              type=int,             default=arch["seed"],                  metavar='S',    help = 'random seed (default: 1)')
    parser.add_argument('--log-interval',      type=int,             default=10,                 metavar='N',    help = 'how many batches to wait before logging training status')
    parser.add_argument('--network-arch',      type=str,             default=arch["network_arch"],           nargs='+',      help = 'Network arch : (default: 256-40-20-1)')
    parser.add_argument('--training-len',      type=int,             default=arch["training_len"],                               help = 'Training len (default: 3500)')
    parser.add_argument('--test-len',          type=int,             default=arch["test_len"],                               help = 'Test len (default: 500)')
    parser.add_argument('--runtime-count',     type=int,             default=0,                                  help = 'this parameter counts that how many times the program is runned')
    parser.add_argument('--display-progress',  action='store_true',  default=False,                              help = 'Display Progress (default:False)')
    parser.add_argument('--data-filename',     type=str,             default="potential-g-20-.dat",              help = 'Data file name to read (default = "potential.dat")')
    parser.add_argument('--label-filename',    type=str,             default="energy-g-20-.dat",                 help = 'Label file name to read (default = "energy.dat")')
    parser.add_argument('--inter-param',       type=float,           default=0.0,                                help = 'Interaction parameter program uses this parameter to choose which file to open (default: 0)')
    parser.add_argument('--test-case',         action='store_true',  default=False,                              help = 'Test Case (default : False) Run-time for test network. No information will be saved.')
    
    return parser