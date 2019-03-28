

def generate_params():

    data_params = {'size_h': 720 // 4,
                   'size_w': 1280 // 4,
                   'size_c': 3,
                   'scale': 4,
                   'train': ['h-y-0306-17'], #['h-y-0306-17', 'h-j-0306-13', 'j-d-0306-21', 'a-y-0305-12', 'd-g-0308-10', 'b-s-0306-19'],
                   'test': ['h-y-0306-17', 'h-j-0306-13', 'j-d-0306-21', 'a-y-0305-12', 'd-g-0308-10', 'b-s-0306-19']}

    train_params = {'batch_size': 8,
                    'lr': 1e-3,
                    'num_episodes': 10000}

    network_params = {'n_resblocks': 10,
                      'kernel_size': [3, 3],
                      'n_feats': 64,
                      'res_scale': 0.1,
                      'shift_mean': True}

    params = {'data': data_params,
              'train': train_params,
              'network': network_params}

    return params
