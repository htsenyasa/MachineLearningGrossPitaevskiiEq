import argparse

def process_parsing(arch, desc = "Artifical Neural Network for Non-linear Schrodinger Equation"):
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--batch-size',        type=int,             default=arch["batch_size"],                 metavar='N',    help='Input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size',   type=int,             default=arch["test_batch_size"],            metavar='N',    help='Input batch size for testing (default: 1000)')
    parser.add_argument('--epochs',            type=int,             default=arch["epoch"],                      metavar='N',    help='Number of epochs to train (default: {})'.format(arch["epoch"]))
    parser.add_argument('--lr',                type=float,           default=arch["lr"],                         metavar='LR',   help='Learning rate (default: {})'.format(arch["lr"]))
    parser.add_argument('--momentum',          type=float,           default=0.2,                                metavar='M',    help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda',           action='store_true',  default=False,                                              help='Disables CUDA training')
    parser.add_argument('--seed',              type=int,             default=arch["seed"],                       metavar='S',    help='Random seed (default: {})'.format(arch["seed"]))
    parser.add_argument('--log-interval',      type=int,             default=10,                                 metavar='N',    help='How many batches to wait before logging training status')
    parser.add_argument('--network-arch',      type=str,             default=arch["network_arch"],               nargs='+',      help='Network arch : (default: {})'.format(arch["network_arch"]))
    parser.add_argument('--training-len',      type=int,             default=arch["training_len"],                               help='Training len (default: {})'.format(arch["training_len"]))
    parser.add_argument('--test-len',          type=int,             default=arch["test_len"],                                   help='Test len (default: {})'.format(arch["test_len"]))
    parser.add_argument('--runtime-count',     type=int,             default=0,                                                  help='This parameter counts that how many times the program is runned')
    parser.add_argument('--display-progress',  action='store_true',  default=False,                                              help='Display Progress (default:False)')
    parser.add_argument('--info-file',         type=str,             default="gaussian",                                         help='Name of file that information will be saved (default: gaussian)')
    parser.add_argument('--data-filename',     type=str,             default="potential-g-20-.dat",                              help='Data file name to read (default = "potential.dat")')
    parser.add_argument('--label-filename',    type=str,             default="energy-g-20-.dat",                                 help='Label file name to read (default = "energy.dat")')
    parser.add_argument('--inter-param',       type=float,           default=0.0,                                                help='Interaction parameter program uses this parameter to choose which file to open (default: 0)')
    parser.add_argument('--test-case',         action='store_true',  default=False,                                              help='Test Case: Run-time for test network. No information will be saved. (default : False)')
    parser.add_argument('--cross-test',        action='store_true',  default=arch["cross_test"],                                 help='Cross Test (Train with A. Test with B). (default : False)')
    parser.add_argument('--data-filename2',    type=str,             default="datafilename2",                                    help='Data file name to read (default = "potential.dat")')
    parser.add_argument('--label-filename2',   type=str,             default="testfilename2",                                    help='Label file name to read (default = "energy.dat")')
    return parser