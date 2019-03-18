import numpy as np


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


def combine_loss(fetches):
    loss = {}
    for key in fetches[0]:
        loss[key] = []

    for i in range(len(fetches)):
        for key in fetches[i]:
            loss[key] += [fetches[i][key]]

    return loss
