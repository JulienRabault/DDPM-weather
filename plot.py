import numpy as np
import matplotlib.pyplot as plt

step=2000
var_names = ['rr', 'u', 'v', 't2m']
for i in range(10):
    sample = np.load(f"sample_step_{step}_{i}.npy")
    for i_var, var in enumerate(var_names):
        plt.clf()
        fig, axes = plt.subplots()
        img = axes.imshow(sample[i_var], origin="lower")
        fig.colorbar(img)
        fig.savefig(f"sample_step_{step}_{var}_{i}.png")
        plt.close()

