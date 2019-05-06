import matplotlib.pyplot as plt
import numpy as np

def compare_image(fake, real):
    dif = np.sum(np.square(fake - real), axis=2)
    plt.matshow((dif), vmin=0, vmax=1)


compare_image()