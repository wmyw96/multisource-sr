import numpy as np
import time
import os


class TimestepSim(object):
    def __init__(self, fps):
        self.total_time = 0
        self.pre_time = 0
        self.fps = fps
        self.status = 0
        self.start()

    def start(self):
        cur_time = time.time()
        if self.status:
            self.total_time += cur_time - self.pre_time
        self.pre_time = time.time()
        self.status = 1

    def get_time(self):
        self.start()
        return int(self.total_time * self.fps)

    def stop(self):
        cur_time = time.time()
        if self.status:
            self.total_time += cur_time - self.pre_time
        self.pre_time = time.time()
        self.status = 0


# noinspection PyTypeChecker
def open_file_and_save(file_path, data):
    """
    :param file_path: type==string
    :param data:
    """
    if os.path.exists(file_path):
        os.remove(file_path)

    # Create the directory for the file
    dir_path = file_path.rsplit('/', 1)[0]
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    with open(file_path, 'wb') as f_handle:
        np.savetxt(f_handle, data, fmt='%s')


def print_metrics(readouts, domain=None):
    '''
    Printing the losses from a sess.run() call
    Args:
        readouts: losses and train_ops : dict
    Returns:
    '''
    spacing = 17
    print_str = '' 
    if domain is not None:
        print_str += domain + '>\n'
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
    print_str += title.rjust(30) + ': '

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


def summarize_loss(fetches):
    loss = {}
    for key in fetches:
        loss[key] = np.mean(fetches[key])

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


def show_variables(domain, myvars):
    print('Trainable Variables in Domain {}: '.format(domain))

    tots = 0
    for var in myvars:
        print(var.name + ': ' + str(var.shape))
        num = 1
        for i in range(len(var.shape)):
            num *= int(var.shape[i])
        tots += num

    tot_b = tots * 4
    tot_kb = tot_b / 1024.0
    tot_mb = tot_kb / 1024.0
    print('Summary: size = {} KB = {} MB'.format(tot_kb, tot_mb))

