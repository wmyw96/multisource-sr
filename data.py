import numpy as np
import os
from skimage import io
from PIL import Image
import progressbar


class dataset(object):
    def __init__(self, inputs, labels, randomize=True):
        self.inputs = inputs
        self.labels = labels
        if len(self.labels.shape) == 1:
            self.labels = np.reshape(self.labels,
                                     [self.labels.shape[0], 1])
        assert len(self.inputs) == len(self.labels)
        self.randomize = randomize
        self.num_pairs = len(inputs)
        self.init_pointer()

    def len(self):
        return self.num_pairs

    def init_pointer(self):
        self.pointer = 0
        if self.randomize:
            idx = np.arange(self.num_pairs)
            np.random.shuffle(idx)
            self.inputs = self.inputs[idx, :]
            self.labels = self.labels[idx, :]

    def get_next_batch(self, batch_size):
        # if batch_size is negative -> return all
        if batch_size < 0:
            return self.inputs, self.labels
        if self.pointer + batch_size >= self.num_pairs:
            self.init_pointer()
        end = self.pointer + batch_size
        inputs = self.inputs[self.pointer:end, :]
        labels = self.labels[self.pointer:end, :]
        self.pointer = end
        return inputs, labels


def sr_dataset(datadir, params):
    # valid = train \belong test
    train_set = set(params['data']['train'])
    lr_img_size = (params['data']['size_w'], params['data']['size_h'])

    train_lr = []
    train_hr = []
    valid_dataset = {}
    test_dataset = {}
    for data_name in params['data']['test']:
        path = os.path.join(datadir, data_name)
        list = os.listdir(path)
        hr_img_cl = []
        lr_img_cl = []

        pgb = progressbar.ProgressBar()
        pgb.start(len(list))
        count = 0

        print('=' * 16 + '\nRead File: {}'.format(path))
        for ele in list:
            if ele[0] == '.':
                continue
            count += 1
            pgb.update(count)

            file_path = os.path.join(path, ele)

            if os.path.isfile(file_path):
                img = io.imread(file_path)
                hr_img_cl.append(np.expand_dims(img, axis=0))

                im = Image.open(file_path)
                shrink = im.resize(lr_img_size, Image.ANTIALIAS)

                lr_img_cl.append(np.expand_dims(shrink, axis=0))
        pgb.finish()

        lr_img_dat = np.concatenate(lr_img_cl, axis=0)
        hr_img_dat = np.concatenate(hr_img_cl, axis=0)
        n_data = lr_img_dat.shape[0]

        if data_name in train_set:
            train_lr.append(lr_img_dat[:n_data // 10 * 6, :])
            train_hr.append(hr_img_dat[:n_data // 10 * 6, :])

            valid_dataset[data_name] = \
                dataset(lr_img_dat[n_data // 10 * 6:n_data // 10 * 8, :],
                        hr_img_dat[n_data // 10 * 6:n_data // 10 * 8, :])
            test_dataset[data_name] = \
                dataset(lr_img_dat[n_data // 10 * 8:, :],
                        hr_img_dat[n_data // 10 * 8:, :])
        else:
            test_dataset[data_name] = \
                dataset(lr_img_dat, hr_img_dat)
    train_lr = np.concatenate(train_lr, axis=0)
    train_hr = np.concatenate(train_hr, axis=0)
    train_dataset = dataset(train_lr, train_hr)

    return train_dataset, valid_dataset, test_dataset
