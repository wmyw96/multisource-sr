import numpy as np
import pandas as pd
import ggplot as gp


class LPTraj(object):
    def __init__(self, name):
        self.name = name
        self.logs = {}

    def record(self, ep, value):
        self.logs[ep] = value

    def to_np_array(self):
        return np.array(sorted(dic.items(), key = lambda k: k[0]))


class LPPlate(object):
    def __init__(self, title):
        self.title = title
        self.trajs = []

    def add_traj(self, traj):
        if isinstance(traj, LPTraj):
            self.trajs.append(traj)
        else:
            raise Exception, "traj is not a valid LPTraj object"

    def plot(self):
        dat = []
        for traj in self.trajs:
            rec = traj.to_np_array()
            rec_len = rec.shape[0]
            label = [traj.name] * rec_len
            lb_array = np.array(label)
            lb_array = np.expand_dims(lb_array, 1)
            dat.append(np.concatenate([rec, lb_array], axis=1))
        df_data = np.concatenate(dat, axis=0)
        df = pd.DataFrame(data=df_data, columns=['ep', 'value', 'type'])
        p = gp.ggplot(gp.aes(x='ep', y='value', color='type'), data=df) + \
            gp.geom_line() + gp.ggtitle(self.title)


def load_lptraj_dict_from_log(log_path):
    
