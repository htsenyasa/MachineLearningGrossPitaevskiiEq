
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
        "test_batch_size": 2000,
        "epoch": 60,
        "lr": 0.001,
        "seed": nlse_seed,
        "network_arch": "CNN",
        "training_len": 8500,
        "test_len" : 1500,
      }


archs = {"FCN": FCN, "CNN": CNN}