def generate_params():

    data_params = {'size_h': 1080 // 4,
                   'size_w': 1920 // 4,
                   'size_c': 3,
                   'scale': 4,
                   'train': ['m3-s'],
                   'test': ['m3-s']}

    train_params = {'batch_size': 16,
                    'lr': 1e-4,
                    'num_episodes': 300000,
                    'disc_steps': 5}

    network_params = {'size_h': 48,
                      'size_w': 48,
                      'size_c': 3,
                      'scale': 4,
                      'n_resblocks': 8,
                      'kernel_size': [3, 3],
                      'n_feats': 32,
                      'res_scale': 0.1,
                      'shift_mean': True}

    disc_params =  {'n_layers': 4,
                    'kernel_size': [4, 4],
                    'n_feats': 32}

    loss_params = {'feat_loss': 0.0,
                   'disc_loss': 50.0,
                   'gp_weight': 10.0}

    params = {'data': data_params,
              'train': train_params,
              'network': network_params,
              'disc': disc_params,
              'loss': loss_params}

    return params

