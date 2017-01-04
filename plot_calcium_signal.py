import matplotlib.pyplot as plt
import numpy as np
import sys


# Configuration for publication-quality plot
## Choose figure size
#figsize = (4, 3) # single-column plot
figsize = (8, 6) # full-column plot
## Choose the font family
#plt.rc('font', family='sans-serif')
plt.rc('font', family='serif')
## Differentiate font size/style between axis labels and tick labels
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
## Use LaTeX
plt.rc('text', usetex=True)

def plot_calcium_signal(input_path, output_path, sampling_rate):
    data = np.genfromtxt(input_path, delimiter=",")
    data = data[1:, :] # remove first row (i.e. column names)
    timesteps = data[:, 0] / sampling_rate
    intensities = data[:, 1]
    plt.figure(figsize=figsize)
    plt.subplot(1, 1, 1)
    plt.plot(timesteps, intensities)
    plt.xlim(np.amin(timesteps), np.amax(timesteps))
    plt.ylim(np.amin(intensities), np.amax(intensities) - 31.5)
    plt.xlabel("time (s)")
    plt.ylabel("intensity (arb. unit)")
    plt.yticks(())
    plt.savefig(output_path, bbox_inches='tight')
    return

if __name__ == '__main__':

    sampling_rate = 20.0 # Hz
    
    argv = sys.argv
    input_path = argv[1]
    output_path = argv[2]
    plot_calcium_signal(input_path, output_path, sampling_rate)
