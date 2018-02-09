
## Common Operations for both CNN and FCN

nlse_seed = 1

FCN = { "batch_size": 30,
        "test_batch_size": 2000,
        "epoch": 60,
        "lr": 0.001,
        "seed": nlse_seed,
        "network_arch": [128, 40, 40, 20, 1],
        "training_len": 8500,
        "test_len" : 1500,
      }


CNN = { "batch_size": 30,
        "test_batch_size": 5500,
        "epoch": 60,
        "lr": 0.001,
        "seed": nlse_seed,  
        "network_arch": "CNN",
        "training_len": 8500,
        "test_len" : 1500,
      }


archs = {"FCN": FCN, "CNN": CNN}

def get_filenames(args):
    if (args.inter_param).is_integer():
        args.inter_param = int(args.inter_param)
    path = ""
    
    #data_filename = "potential-g-{}-.dat".format(args.inter_param)
    #label_filename = "energy-g-{}-.dat".format(args.inter_param)
    data_filename = "gp_ml_dataset/var_g_var_pot/potential-var_g_var_pot_.dat"
    label_filename = "gp_ml_dataset/var_g_var_pot/energy-var_g_var_pot_.dat"
    #data_filename = "potential-g-{}-.dat".format(args.inter_param)
    #label_filename = "mergeddata/eint_kin_pot_energy-g-{}-.dat".format(args.inter_param)

    return data_filename, label_filename
