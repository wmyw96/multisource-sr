

def generate_params():

    data_params = {'size_h': 728 // 2,
                   'size_w': 1024 // 2,
                   'size_c': 3,
                   'scale': 2,
                   'train': ['h-y-0306-13', 'j-d-0306-21'],
                   'test': ['h-y-0306-13', 'h-j-0306-13', 'j-d-0306-21']}

    train_params = {'batch_size': 16,
                    'lr': 1e-4,
                    'num_episodes': 1000}

    network_params = {'n_resblocks': 16,
                      'kernel_size': [3, 3],
                      'n_feats': 64,
                      'res_scale': 0.1,
                      'shift_mean': True}

    params = {'data': data_params,
              'train': train_params,
              'network': network_params}

    return params
