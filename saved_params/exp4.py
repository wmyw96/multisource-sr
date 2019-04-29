
def generate_params():

    data_params = {'size_h': 720 // 2,
                   'size_w': 1280 // 2,
                   'size_c': 3,
                   'scale': 2,
                   'train': ['c-x-0306-18'],
                   'test': ['c-x-0306-18']}

    train_params = {'batch_size': 16,
                    'lr': 1e-3,
                    'num_episodes': 300000,
                    'trainlog_interval': 1000,
                    'decay_interval': 50000,
                    'validlog_interval': 5000}

    network_params = {'size_h': 48,
                      'size_w': 48,
                      'size_c': 3,
                      'scale': 2,
                      'n_resblocks': 16,
                      'kernel_size': [3, 3],
                      'n_feats': 64,
                      'res_scale': 0.1,
                      'shift_mean': True,
                      'start_branch': 8,
                      'n_feats_branch': 32,
                      'n_resblocks_branch': 4,
                      'n_sources': 7}

    params = {'data': data_params,
              'train': train_params,
              'network': network_params}

    return params


