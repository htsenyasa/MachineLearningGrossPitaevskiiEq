
## Common Operations for both CNN and FCN

nlse_seed = 11

FCN = { "batch_size": 30,
        "test_batch_size": 2000,
        "epoch": 60,
        "lr": 0.001,
        "seed": nlse_seed,
        "network_arch": [128, 40, 40, 20, 1],
        "training_len": 8500,
        "test_len" : 1500,
        "cross_test" : False,        
      }


CNN = { "batch_size": 50,
        "test_batch_size": 25,
        "epoch": 10,
        "lr": 0.003,
        "seed": nlse_seed,  
        "network_arch": "CNN",
        "training_len": 40000,
        "test_len" : 5000,
        "cross_test" : False,
      }


archs = {"FCN": FCN, "CNN": CNN}

data_filename =  "generic_dataset_MAIN/random/pot_inter.dat.npy"
label_filename = "generic_dataset_MAIN/random/energy-generic.dat"


def get_filenames(args):
    if (args.inter_param).is_integer():
        args.inter_param = int(args.inter_param)
    path = ""
    
    #data_filename = "potential-g-{}-.dat".format(args.inter_param)
    #label_filename = "energy-g-{}-.dat".format(args.inter_param)
    
    #data_filename =  "gp_ml_dataset/var_g/potential_inter.dat"
    #label_filename = "gp_ml_dataset/var_g/energy-var_g.dat"
    
    #data_filename =  "gp_ml_dataset/var_g_var_pot/pot_inter_merge2.dat"
    #label_filename = "gp_ml_dataset/var_g_var_pot/energy-var_g_var_pot_.dat"

    #data_filename  = "pot_inter.dat.npy"
    #label_filename = "energy-generic.dat.npy"
    
    #data_filename = "generic_dataset_2d/harmonic/potential.h5"
    #label_filename = "generic_dataset_2d/harmonic/features.h5"

    #data_filename =  "generic_dataset_MAIN/gaussian/pot_inter.dat.npy"
    #label_filename = "generic_dataset_MAIN/gaussian/energy-generic.dat"

    #data_filename =  "/run/media/user/TOSHIBA/gp_data/var_g_var_freq/potential-var_g_var_freq_.dat"
    #label_filename = "/run/media/user/TOSHIBA/gp_data/var_g_var_freq/energy-var_g_var_freq_.dat"
    
    #data_filename =  "gp_ml_dataset/var_g_var_pot/potential-var_g_var_pot_.dat"
    #label_filename = "gp_ml_dataset/var_g_var_pot/energy-var_g_var_pot_.dat"
    
    #data_filename = "potential-g-{}-.dat".format(args.inter_param)
    #label_filename = "mergeddata/eint_kin_pot_energy-g-{}-.dat".format(args.inter_param)

    return data_filename, label_filename
