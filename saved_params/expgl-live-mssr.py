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
                      'n_feats': 46,
                      'res_scale': 0.1,
                      'shift_mean': True,
                      'shared_block': 4,
                      'n_feats_branch': 12,
                      'n_resblocks_branch_b': 4,
                      'n_resblocks_branch_a': 4,
                      'n_sources': 6}

    live_params = {'fps': 2,
                   'basefile': ['h-y-0306-17', 'h-j-0306-13', 'j-d-0306-21', 'd-g-0308-10', 'b-s-0306-19', 'a-y-0305-12'],
                   'base_name': 'h-y-0306-17'}
    
    params = {'data': data_params,
              'train': train_params,
              'network': network_params,
              'live': live_params}

    return params