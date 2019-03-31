import numpy as np
import time


def print_metrics(readouts):
    '''
    Printing the losses from a sess.run() call
    Args:
        readouts: losses and train_ops : dict
    Returns:
    '''
    spacing = 17
    print_str = ''
    for k_, v_ in readouts.items():
        if 'loss' in k_:
            value = np.around(np.mean(v_, axis=0), decimals=6)
            print_str += (k_ + ': ').rjust(spacing) + str(value) + '\n'

    print_str = print_str[:-2]
    print(print_str)


def write_logs(title, readouts, path):
    '''
    Printing the losses from a sess.run() call
    Args:
        readouts: losses and train_ops : dict
    Returns:
    '''

    print_str = ''
    print_str += title.rjust(20) + ': '

    spacing = 10
    for k_, v_ in readouts.items():
        if 'loss' in k_:
            value = np.around(np.mean(v_, axis=0), decimals=6)
            print_str += (k_ + ' ').rjust(spacing) + str(value) + ','

    print_str = print_str[:-1]

    with open(path, 'a') as f:
        f.write(print_str + '\n')


def summarize_loss(fetches):
    loss = {}
    for key in fetches:
        loss[key] = np.mean(fetches[key])

    return loss


def combine_loss(fetches):
    loss = {}
    for key in fetches[0]:
        loss[key] = []

    for i in range(len(fetches)):
        for key in fetches[i]:
            loss[key] += [fetches[i][key]]

    return loss


def get_log_name(params, model):
    ts = (time.strftime("%m%d%H", time.localtime()))
    scale = 'sl%d' % (params['data']['scale'])
    train = 'tr%d' % (len(params['data']['train']))
    #if len(params['data']['train']) == 1:
    #    train += '_' + params['data']['train'][0][0:3]
    network = 'netb%df%d' % (params['network']['n_resblocks'],
                             params['network']['n_feats'])
    return model + '_' + ts + '_' + scale + '_' + train + '_' + network

