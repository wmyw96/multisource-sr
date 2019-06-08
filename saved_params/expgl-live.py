def generate_params():

    data_params = {'size_h': 720 // 4,
                   'size_w': 1280 // 4,
                   'size_c': 3,
                   'scale': 4,
                   'train': ['hy030613'],
                   'test': ['hy030613']}

    train_params = {'batch_size': 16,
                    'lr': 5 * 1e-4,
                    'num_episodes': 300000}

    network_params = {'size_h': 48,
                      'size_w': 48,
                      'size_c': 3,
                      'scale': 4,
                      'n_resblocks': 8,
                      'kernel_size': [3, 3],
                      'n_feats': 48,
                      'res_scale': 0.1,
                      'shift_mean': True}

    live_params = {'fps': 2}
    
    params = {'data': data_params,
              'train': train_params,
              'network': network_params,
              'live': live_params}

    return params