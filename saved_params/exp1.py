

def generate_params():

    data_params = {'size_h': 720 // 2,
                   'size_w': 1280 // 2,
                   'size_c': 3,
                   'scale': 2,
                   'train': ['h-y-0306-17'],
                   'test': ['h-y-0306-17']}

    train_params = {'batch_size': 16,
                    'lr': 1e-4,
                    'num_episodes': 300000}

    network_params = {'size_h': 48,
                      'size_w': 48,
                      'size_c': 3,
                      'scale': 2,
                      'n_resblocks': 16,
                      'kernel_size': [3, 3],
                      'n_feats': 64,
                      'res_scale': 0.1,
                      'shift_mean': True}

    params = {'data': data_params,
              'train': train_params,
              'network': network_params}

    return params
