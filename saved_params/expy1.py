def generate_params():

    data_params = {'size_h': 1080 // 4,
                   'size_w': 1920 // 4,
                   'size_c': 3,
                   'scale': 4,
                   'train': ['g8'],
                   'test': ['g8']}

    train_params = {'batch_size': 16,
                    'lr': 1e-4,
                    'num_episodes': 300000}

    network_params = {'size_h': 48,
                      'size_w': 48,
                      'size_c': 3,
                      'scale': 4,
                      'n_resblocks': 8,
                      'kernel_size': [3, 3],
                      'n_feats': 32,
                      'res_scale': 0.1,
                      'shift_mean': True}

    params = {'data': data_params,
              'train': train_params,
              'network': network_params}

    return params

