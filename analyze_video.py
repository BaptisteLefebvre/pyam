################################################################################
# """
# The program 'eulerian.py' reproduces the results of the paper Eulerian Video
# Magnification for Revealing Subtle Changes in the World by Hao-Yu Wu, Michael
# Rubinstein, Eugene Shih, John Guttag, Fredo Durand and William T. Freeman from
# the MIT CSAIL and Quanta Research Cambridge, Inc (SIGGRAPH 2012).
# """

# # Author: Baptiste Lefebvre <baptiste.lefebvre@ens.fr>
# # License: MIT



# import cv2 as cv
# import math as mt
# import numpy as np
# import os
# import scipy as sp
# import scipy.misc



# def fileparts(file):
#     """
#     Returns the directory, file name and file name extension for the specified
#     file.
#     """
#     directory, base = os.path.split(file)
#     name, extension = os.path.splitext(base)
#     return (directory, name, extension)

# def fullfile(directory, name, extension):
#     """
#     Builds a full file specification from the directory, file name and file name
#     extension specified.
#     """
#     base = name + extension
#     return os.path.join(directory, base)

# def im2double(image):
#     """
#     Convert image to double precision.
#     """
#     return image.astype(np.float64)

# def rgb2yiq(rgb_frame):
#     """
#     Change the color space of an image from RGB to YIQ.
#     """
#     shape = rgb_frame.shape
#     M = np.array([[+0.299, +0.596, +0.211],
#                   [+0.587, -0.274, -0.523],
#                   [+0.114, -0.322, +0.312]])
#     yiq_frame = np.zeros(shape, dtype=np.float64)
#     for i in xrange(0, shape[0]):
#         yiq_frame[i, :, :] = np.dot(rgb_frame[i, :, :], M)
#     return yiq_frame

# def blurDn(input_frame, level):
#     """
#     Blur and downsampling an image.
#     """
#     if 1 < level:
#         output_frame = blurDn(cv.pyrDown(input_frame), level - 1)
#     else:
#         output_frame = cv.pyrDown(input_frame)
#     return output_frame

# def blurDnClr(input_frame, level):
#     """
#     Blur and downsampling a 3-color image.
#     """
#     temp = blurDn(input_frame[:, :, 0], level)
#     shape = (temp.shape[0], temp.shape[1], input_frame.shape[2])
#     output_frame = np.zeros(shape, dtype=np.float64)
#     output_frame[:, :, 0] = temp
#     for i in xrange(1, input_frame.shape[2]):
#         output_frame[:, :, i] = blurDn(input_frame[:, :, i], level)
#     return output_frame

# def build_Gdown_stack(input_file, start_index, end_index, level):
#     """
#     Apply Gaussian pyramid decomposition on the input file from the start index
#     to the end index and select a specific band indicated by level.
#     """
#     # Read video.
#     input_video = cv.VideoCapture(input_file)
#     # Extract video info.
#     video_width = int(input_video.get(cv.CAP_PROP_FRAME_WIDTH))
#     video_height = int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))
#     video_number_channels = 3
#     _, rgb_frame = input_video.read()
    
#     # First frame.
#     rgb_frame = im2double(rgb_frame)
#     yiq_frame = rgb2yiq(rgb_frame)
    
#     blurred_image = blurDnClr(yiq_frame, level)
    
#     # Create pyramidal stack.
#     shape = (end_index - start_index + 1,
#              blurred_image.shape[0],
#              blurred_image.shape[1],
#              blurred_image.shape[2])
#     Gdown_stack = np.zeros(shape, dtype=np.float64)
#     Gdown_stack[0, :, :, :] = blurred_image
#     for k in xrange(1 + start_index, 1 + end_index):
#         _, rgb_frame = input_video.read()
#         rgb_frame = im2double(rgb_frame)
#         yiq_frame = rgb2yiq(rgb_frame)
#         blurred_frame = blurDnClr(yiq_frame, level)
#         Gdown_stack[k, :, :, :] = blurred_frame
#     return Gdown_stack

# def ideal_bandpassing(input, dimension, wl, wh, sampling_rate):
#     """
#     Apply ideal band pass filter on the input along the specified dimension.
#     """
#     input_shifted = np.rollaxis(input, dimension - 1)
#     shape = input_shifted.shape
    
#     n = shape[0]
#     dn = len(shape)
    
#     freq = np.arange(0, n, dtype=np.float)
#     freq = freq / float(n) * float(sampling_rate)
#     mask = np.logical_and(wl < freq, freq < wh)
#     mask = np.reshape(mask, (mask.shape[0],) + (1,) * (len(shape) - 1))
    
#     shape = (1,) + shape[1:];
#     mask = np.tile(mask, shape)
    
#     F = np.fft.fft(input_shifted, axis=0)
#     F[np.logical_not(mask)] = 0
#     output = np.fft.ifft(F, axis=0).real
#     output = np.rollaxis(output, (dn - (dimension - 1)) % dn)
#     return output

# def imresize(frame, size):
#     """
#     Resize an image.
#     """
#     #return sp.misc.imresize(frame, size)
#     return cv.resize(frame, size)

# def yiq2rgb(yiq_frame):
#     """
#     Change the color space of an image from YIQ to RGB.
#     """
#     shape = yiq_frame.shape
#     M = np.array([[+1.000, +1.000, +1.000],
#                   [+0.956, -0.272, -1.106],
#                   [+0.621, -0.647, +1.703]])
#     rgb_frame = np.zeros(shape, dtype=np.float64)
#     for i in xrange(0, shape[0]):
#         rgb_frame[i, :, :] = np.dot(yiq_frame[i, :, :], M)
#     return rgb_frame

# def im2uint8(frame):
#     """
#     Convert image to 8-bit unsigned integers.
#     """
#     return frame.astype(np.uint8)

# def amplify_spatial_Gdown_temporal_ideal(input_file, output_directory, alpha,
#                                          level, fl, fh, sampling_rate,
#                                          chrom_attenuation):
#     """
#     TODO: add description.
#     """
#     _, input_name, _ = fileparts(input_file)
#     output_name = input_name \
#                   + "-ideal-from-" + repr(fl) \
#                   + "-to-" + repr(fh) \
#                   + "-alpha-" + repr(alpha) \
#                   + "-level-" + repr(level) \
#                   + "-chromAtn-" + repr(chrom_attenuation)
#     output_extension = ".avi"
#     output_file = fullfile(output_directory, output_name, output_extension)
    
#     # Read video.
#     input_video = cv.VideoCapture(input_file)
#     # Extract video info.
#     video_width = int(input_video.get(cv.CAP_PROP_FRAME_WIDTH))
#     video_height = int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))
#     video_number_channels = 3
#     video_frame_rate = input_video.get(cv.CAP_PROP_FPS)
#     if mt.isnan(video_frame_rate):
#         video_frame_rate = 25.0
#     video_length = int(input_video.get(cv.CAP_PROP_FRAME_COUNT))
#     # Display video info.
#     print("width: {0}" \
#           "\nheight: {1}" \
#           "\nnumber of channels: {2}" \
#           "\nframe rate: {3}" \
#           "\nlength: {4}".format(video_width,
#                                  video_height,
#                                  video_number_channels,
#                                  video_frame_rate,
#                                  video_length))
#     temp = None
    
#     start_index = 1 - 1
#     end_index = video_length - 10 - 1; # TODO: understand why -10.
    
#     video_fourcc = cv.VideoWriter_fourcc(*'FMP4')
#     video_size = (video_width, video_height)
#     output_video = cv.VideoWriter(output_file, video_fourcc, video_frame_rate, video_size)
    
#     # Compute Gaussian blur stack.
#     print("Spatial filtering...")
#     Gdown_stack = build_Gdown_stack(input_file, start_index, end_index, level)
#     print("Finished")
    
#     # Temporal filtering.
#     print("Temporal filtering...")
#     filtered_stack = ideal_bandpassing(Gdown_stack, 1, fl, fh, sampling_rate)
#     print("Finished")

#     # Amplify.
#     filtered_stack[:, :, :, 0] = filtered_stack[:, :, :, 0] * alpha
#     filtered_stack[:, :, :, 1] = filtered_stack[:, :, :, 1] * alpha * chrom_attenuation
#     filtered_stack[:, :, :, 2] = filtered_stack[:, :, :, 2] * alpha * chrom_attenuation
    
#     # Render on the input video.
#     print("Rendering...")
#     for k in xrange(start_index, end_index + 1):
#         _, rgb_frame = input_video.read()
#         rgb_frame = im2double(rgb_frame)
#         yiq_frame = rgb2yiq(rgb_frame)

#         filtered_frame = filtered_stack[k, :, :, :]
#         filtered_frame = imresize(filtered_frame, (video_width, video_height))
#         filtered_frame = yiq_frame + filtered_frame
        
#         output_frame = yiq2rgb(filtered_frame)
#         output_frame[output_frame < 0] = 0
#         output_frame[255 < output_frame] = 255
#         output_frame = im2uint8(output_frame)
#         output_video.write(output_frame)
#     print("Finished")
    
#     # Release everything.
#     input_video.release()
#     output_video.release()
#     cv.destroyAllWindows()

################################################################################
# # Plot Fast Fourier Transform
# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.fftpack

# # Number of samplepoints
# N = 600
# # sample spacing
# T = 1.0 / 800.0
# x = np.linspace(0.0, N*T, N)
# y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
# yf = scipy.fftpack.fft(y)
# xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

# fig, ax = plt.subplots()
# ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
# plt.show()
################################################################################
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL as pil
import PIL.Image
import sys
import tifffile as tf


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

plt.rcParams['image.cmap'] = 'viridis'
plt.rcParams['image.interpolation'] = 'nearest'


num_ = 512 # number of sample points for highpass/lowpass transition profile


def load_tiff(input_path):
    # Open the TIFF file with PIL.
    video = pil.Image.open(input_path)
    # Retrieve the number of frames, width and height.
    n_frames = video.n_frames
    width = video.width
    height = video.height
    # Retrieve the different frames.
    shape = (n_frames, height, width)
    dtype = 'uint8'
    frames = np.empty(shape, dtype=dtype)
    for index in range(0, n_frames):
        # TODO: add a progress bar...
        video.seek(index)
        frame = np.array(video)
        if len(frame.shape) == 3:
            frames[index, :, :] = frame[:, :, 0]
        else:
            frames[index, :, :] = frame[:, :]
    # Return the frames.
    return frames


def save_tiff(frames, output_path):
    frames = frames.astype('uint8')
    tf.imsave(output_path, frames, imagej=True)
    return


def save_avi(frames, path, fourcc="MJPG", fps=50.0):
    fourcc = cv2.VideoWriter_fourcc(*fourcc)
    width = frames.shape[2]
    height = frames.shape[1]
    size = (width, height)
    video = cv2.VideoWriter(path, fourcc, fps, size)
    n_frames = frames.shape[0]
    for index in range(n_frames):
        # TODO: add a progress bar...
        frame = frames[index, :, :]
        frame = frame.astype('uint8')
        frame = np.dstack((frame, frame, frame)) # recompose the three RGB layer
        video.write(frame)
    video.release()
    return


def blur_down(input_frame, level):
    """
    Blur and downsampling a frame.
    """
    if 1 < level:
        output_frame = blur_down(cv2.pyrDown(input_frame), level - 1)
    else:
        output_frame = cv2.pyrDown(input_frame)
    return output_frame


def build_gaussian_down_stack(input_path, start_index, end_index, level):
    """
    Apply Gaussian pyramid decomposition on the input file from the start index
    to the end index and select a specific band indicated by level.
    """
    # Read video
    frames = load_tiff(input_path)
    
    # Extract video info
    n_frames = frames.shape[0]
    height = frames.shape[1]
    width = frames.shape[2]

    if start_index is None:
        start_index = 0
    if end_index is None:
        end_index = n_frames - 1
    
    # First frame
    frame = frames[start_index, :, :]
    frame = frame.astype('float')
    
    blurred_frame = blur_down(frame, level)
    
    # Create pyramidal stack.
    shape = (end_index - start_index + 1,
             blurred_frame.shape[0],
             blurred_frame.shape[1])
    gaussian_down_stack = np.zeros(shape, dtype='float')
    gaussian_down_stack[0, :, :] = blurred_frame
    for current_index in range(start_index + 1, end_index + 1):
        frame = frames[current_index, :, :]
        frame = frame.astype('float')
        blurred_frame = blur_down(frame, level)
        gaussian_down_stack[current_index, :, :] = blurred_frame
    return gaussian_down_stack


def ideal_bandpassing(input_stack, dim, fl, fh, sr):
    """
    Apply ideal band pass filter on the input along the specified dimension.
    fl: low frequency
    fh: high frequency
    sr: sampling rate
    """
    rolled_stack = np.rollaxis(input_stack, dim, 0) # set ft dimension to 0
    shape = rolled_stack.shape
    
    n = shape[0]
    dn = len(shape)
    
    freqs = np.arange(0, n, dtype='float')
    freqs = freqs / float(n) * float(sr)
    mask = np.logical_and(fl <= freqs, freqs < fh)
    shape_tmp = (n,) + (1,) * (len(shape) - 1)
    mask = np.reshape(mask, shape_tmp)

    shape_tmp = (1,) + shape[1:]
    mask = np.tile(mask, shape_tmp)
    
    ft_stack = np.fft.fft(rolled_stack, axis=dim)
    ft_stack[np.logical_not(mask)] = 0.0
    output_stack = np.fft.ifft(ft_stack, axis=dim).real
    output_stack = np.rollaxis(output_stack, 0, dim)
    
    return output_stack


def plot_frequencies(input_stack, fs, i=None, j=None):
    """
    fs: sampling frequency
    """

    n_frames = input_stack.shape[0]
    if i is None:
        height = input_stack.shape[1]
        i = np.random.choice(height)
    if j is None:
        width = input_stack.shape[2]
        j = np.random.choice(width)
    s = input_stack[:, i, j]
    
    # nfft = 512
    # noverlap = 256 + 128 + 64 + 32 + 16 + 8
    # nfft = 256
    # noverlap = 128 + 64 + 32 + 16 + 8 + 4
    nfft = 128
    noverlap = 64 + 32 + 16 + 8 + 4
    # nfft = 64
    # noverlap = 32 + 16 + 8 + 4
    
    ndiff = nfft - noverlap

    xmin = 0.0
    xmax = float(n_frames) / fs
    t = np.linspace(xmin, xmax, n_frames)
    
    plt.figure(figsize=figsize)
    plt.subplot(1, 1, 1)
    plt.plot(t, s)
    plt.grid()
    plt.xlabel("time (s)")
    plt.ylabel("intensity (arb. unit)")
    plt.xlim(xmin, xmax)
    plt.tight_layout()

    xmin = (float(nfft / 2 + 0) - 0.5 * float(ndiff)) / fs
    xmax = (float(nfft / 2 + ((n_frames - nfft) / ndiff) * ndiff) + 0.5 * float(ndiff)) / fs
    xextent = (xmin, xmax)
    
    plt.figure(figsize=figsize)
    plt.subplot(1, 1, 1)
    spectrum, freqs, t, im = plt.specgram(s,
                                          NFFT=nfft,
                                          Fs=fs,
                                          noverlap=noverlap,
                                          detrend='default',
                                          scale='dB',
                                          xextent=xextent,
                                          cmap='viridis',
                                          interpolation='nearest',
                                          aspect='auto')
    plt.xlim(xmin, xmax)
    # plt.title("Spectrogram of pixel ({}, {})".format(i, j))
    plt.xlabel("time (s)")
    plt.ylabel("frequency (Hz)")
    cbar = plt.colorbar(im)
    cbar.set_label("amplitude (dB)")
    plt.tight_layout()
    
    return


def find_period(timestep_deltas):
    deltas = timestep_deltas[1 < timestep_deltas]
    period = np.amin(deltas)
    threshold_min = period - period / 2
    threshold_max = period + period / 2
    deltas = deltas[threshold_min <= deltas]
    deltas = deltas[deltas <= threshold_max]
    period = np.mean(deltas)
    return period


def find_missing_flashes(timesteps, timestep_deltas, timestep_period):
    missing_timesteps = []
    for i, timestep_delta in enumerate(timestep_deltas):
        if 1.5 * timestep_period <= timestep_delta:
            n = int(np.round(float(timestep_delta) / timestep_period))
            tmp_timesteps = np.linspace(timesteps[i], timesteps[i+1], n + 1)
            tmp_timesteps = tmp_timesteps[1:-1]
            tmp_timesteps = np.around(tmp_timesteps)
            tmp_timesteps = tmp_timesteps.astype('int')
            missing_timesteps.append(tmp_timesteps)
    timesteps = np.concatenate((timesteps,) + tuple(missing_timesteps))
    timestaps = np.unique(timesteps)
    timestaps = np.sort(timesteps)
    return timesteps


def print_flash_times(input_stack, fs):
    flags = (input_stack == 182.0)
    flags = np.max(flags, axis=(1, 2))
    timesteps = np.nonzero(flags)[0]
    timestep_deltas = np.diff(timesteps)
    timestep_period = find_period(timestep_deltas)
    timesteps = find_missing_flashes(timesteps, timestep_deltas, timestep_period)
    # timesteps = tuple([timesteps - offset for offset in [-2, -1, 0, +1, +2]])
    # timesteps = np.concatenate(timesteps)
    # timesteps = np.unique(timesteps)
    # timesteps = np.sort(timesteps)
    timestep_deltas = np.diff(timesteps)
    times = timesteps.astype('float') / fs
    time_deltas = np.diff(times)
    time_period = float(timestep_period) / fs
    print("Flash times:")
    print("  timesteps      : {}".format(timesteps))
    print("  timestep deltas: {}".format(timestep_deltas))
    print("  timestep period: {}".format(timestep_period))
    print("  times          : {}".format(times))
    print("  time deltas    : {}".format(time_deltas))
    print("  time period    : {}".format(time_period))
    return

def reconstruct_flashes(input_stack):
    flags = (input_stack == 182.0)
    flags = np.max(flags, axis=(1, 2))
    timesteps = np.nonzero(flags)[0]
    timestep_deltas = np.diff(timesteps)
    timestep_period = find_period(timestep_deltas)
    timesteps = find_missing_flashes(timesteps, timestep_deltas, timestep_period)
    flags[timesteps] = True
    flags[timesteps - 2] = True
    flags[timesteps - 1] = True
    flags[timesteps + 1] = True
    flags[timesteps + 2] = True
    flags = np.logical_not(flags)
    n_frames = input_stack.shape[0]
    timesteps = np.arange(0, n_frames, dtype='float')
    output_stack = np.empty(input_stack.shape)
    height = input_stack.shape[1]
    width = input_stack.shape[2]
    for i in range(0, height):
        for j in range(0, width):
            output_stack[:, i, j] = np.interp(timesteps, timesteps[flags], input_stack[flags, i, j])
    return output_stack


def analyze_spatial_gaussian_down_temporal_ideal(input_path, output_path):
    
    start_index = None
    end_index = None
    level = 3
    
    # Compute Gaussian blur stack.
    print("Spatial filtering...")
    gaussian_down_stack = build_gaussian_down_stack(input_path, start_index, end_index, level)
    print("Done.")
    # TODO: clean the following lines
    # save_avi(gaussian_down_stack, output_path, fourcc="MJPG", fps=100.0)
    # save_avi(gaussian_down_stack, output_path, fourcc="X264", fps=100.0)
    # save_tiff(gaussian_down_stack, output_path)
    
    dim = 0
    fl = 10.0 # Hz # low frequency # TODO: correct
    fh = 50.0 # Hz # high frequency # TODO: correct
    sr = 20.0 # Hz # sampling rate # TODO: correct
    
    # # Temporal filtering.
    # print("Temporal filtering...")
    # filtered_stack = ideal_bandpassing(gaussian_down_stack, dim, fl, fh, sampling_rate)
    # print("Done.")

    print("Pixel choice...")
    height = gaussian_down_stack.shape[1]
    width = gaussian_down_stack.shape[2]
    i = np.random.choice(height)
    j = np.random.choice(width)
    print("  i: {}".format(i))
    print("  j: {}".format(j))
    print("Done.")
    
    # Temporal analysis
    print("Temporal analysis...")
    plot_frequencies(gaussian_down_stack, sr, i=i, j=j)
    print("Done.")
    
    # Reconstruct missing data
    print("Reconstruct missing data...")
    print_flash_times(gaussian_down_stack, sr)
    clean_stack = reconstruct_flashes(gaussian_down_stack)
    print("Done.")
    save_tiff(clean_stack, output_path)
    
    # Second temporal analysis
    print("Temporal analysis...")
    plot_frequencies(clean_stack, sr, i=i, j=j)
    print("Done.")

    # sys.exit()
    plt.show()
    
    return


def arange(xmin, xmax):
    x = np.arange(xmin, xmax + 1)
    return x

def point_operator(a, x, y):
    shape = a.shape
    a = a.flatten()
    a = np.interp(a, x, y)
    a = np.reshape(a, shape)
    return a

def design_highpass_filter(h, w, l, n_o, num=num_):
    
    if h % 2 == 0:
        hmin = - h / 2
        hmax = h / 2 - 1
    else:
        hmin = - (h - 1) / 2
        hmax = (h - 1) / 2
    if w % 2 == 0:
        wmin = - w / 2
        wmax = w / 2 - 1
    else:
        wmin = (w - 1) / 2
        wmax = - (w - 1) / 2
    xlim = min(max(- hmin, hmax), max(-wmin, wmax))
    # xlow = np.power(0.5, n_o * (l + 1) + 1)
    xlow = np.power(0.5, n_o * 1.0 + 1.0)
    # xhigh = np.power(0.5, n_o * l + 1)
    xhigh = np.power(0.5, n_o * 0.0 + 1.0)
    xmmin = xlow * xlim
    xmmax = xhigh * xlim
    xm = np.linspace(xmmin, xmmax, num=num)
    # ym = np.cos(0.5 * np.pi + 0.5 / n_o * np.pi * np.log2(2 ** (n_o * (l + 1) + 1) * xm / xlim))
    ym = np.cos(0.5 * np.pi + 0.5 / n_o * np.pi * np.log2(2 ** (n_o * 1.0 + 1.0) * xm / xlim))
    ym = np.power(ym, 2.0)
    
    h = np.concatenate((arange(0, hmax), arange(hmin, -1)))
    w = np.concatenate((arange(0, hmax), arange(hmin, -1)))
    h2, w2 = np.meshgrid(h, w)
    magnitude = np.sqrt(np.power(h2, 2.0) + np.power(w2, 2.0))
    angle = np.arctan2(h2, w2)
    
    highpass_filter = point_operator(magnitude, xm, ym)
    
    return highpass_filter

def design_lowpass_filter(h, w, l, b, n_o, num=num_):
    
    if h % 2 == 0:
        hmin = - h / 2
        hmax = h / 2 - 1
    else:
        hmin = - (h - 1) / 2
        hmax = (h - 1) / 2
    if w % 2 == 0:
        wmin = - w / 2
        wmax = w / 2 - 1
    else:
        wmin = (w - 1) / 2
        wmax = - (w - 1) / 2
    xlim = min(max(-hmin, hmax), max(-wmin, wmax))
    # xlow = np.power(0.5, n_o * l + 1)
    xlow = np.power(0.5, n_o * (1.0 / n_o - b) + 1.0)
    # xhigh = np.power(0.5, n_o * (l - 1) + 1)
    xhigh = np.power(0.5, n_o * (1.0 / n_o - 1.0 - b) + 1.0)
    xmmin = xlow * xlim
    xmmax = xhigh * xlim
    xm = np.linspace(xmmin, xmmax, num=num)
    # ym = np.cos(0.5 / n_o * np.pi * np.log2(2 ** (n_o * l + 1) * xm / xlim))
    ym = np.cos(0.5 / n_o * np.pi * np.log2(2 ** (n_o * (1.0 / n_o - b) + 1.0) * xm / xlim))
    ym = np.power(ym, 2.0)
    
    h = np.concatenate((arange(0, hmax), arange(hmin, -1)))
    w = np.concatenate((arange(0, hmax), arange(hmin, -1)))
    h2, w2 = np.meshgrid(h, w)
    magnitude = np.sqrt(np.power(h2, 2.0) + np.power(w2, 2.0))
    angle = np.arctan2(h2, w2)
    
    lowpass_filter = point_operator(magnitude, xm, ym)
    
    return lowpass_filter

def design_bandpass_filter(h, w, l, d, b, n_d, n_o, num=num_):
    """
    h: image height
    w: image width
    d: direction
    b: band
    n_d: number of directions
    n_o: number of octaves
    """
    
    if h % 2 == 0:
        hmin = - h / 2
        hmax = h / 2 - 1
    else:
        hmin = - (h - 1) / 2
        hmax = (h - 1) / 2
    if w % 2 == 0:
        wmin = - w / 2
        wmax = w / 2 - 1
    else:
        wmin = (w - 1) / 2
        wmax = - (w - 1) / 2
    xlim = min(max(-hmin, hmax), max(-wmin, wmax))
    # xlow = np.power(0.5, n_o * (l + 1) + 1)
    xlow = np.power(0.5, n_o * (1.0 / n_o + 1.0 - b) + 1.0)
    # xmid = np.power(0.5, n_o * l + 1)
    xmid = np.power(0.5, n_o * (1.0 / n_o - b) + 1.0)
    # xhigh = np.power(0.5, n_o * (l - 1) + 1)
    xhigh = np.power(0.5, n_o * (1.0 / n_o - 1.0 - b) + 1.0)
    xm1min = xlow * xlim
    xm1max = xmid * xlim
    xm2min = xmid * xlim
    xm2max = xhigh * xlim
    xm1 = np.linspace(xm1min, xm1max, num=num)
    # ym1 = np.cos(0.5 * np.pi + 0.5 / n_o * np.pi * np.log2(2 ** (n_o * (l + 1) + 1) * xm1 / xlim))
    ym1 = np.cos(0.5 * np.pi + 0.5 / n_o * np.pi * np.log2(2 ** (n_o * (1.0 / n_o + 1.0 - b) + 1.0) * xm1 / xlim))
    ym1 = np.power(ym1, 2.0)
    xm2 = np.linspace(xm2min, xm2max, num=num)
    # ym2 = np.cos(0.5 / n_o * np.pi * np.log2(2 ** (n_o * l + 1) * xm2 / xlim))
    ym2 = np.cos(0.5 / n_o * np.pi * np.log2(2 ** (n_o * (1.0 / n_o - b) + 1.0) * xm2 / xlim))
    ym2 = np.power(ym2, 2.0)
    xm = np.concatenate((xm1, xm2))
    ym = np.concatenate((ym1, ym2))
    
    xamin = - (num - 1) / num * np.pi
    xamax = np.pi
    xa = np.linspace(xamin, xamax, num=num)
    ya = xa - d / n_d * np.pi
    ya = np.cos(ya)
    ya[ya < 0.0] = 0.0
    ya = np.power(ya, 2.0 * (n_d - 1.0))
    alpha = np.sum(np.power(np.cos(0.5 * np.pi - np.arange(0, n_d) / n_d * np.pi), 2.0 * (n_d - 1.0)))
    ya = ya / alpha
    
    h = np.concatenate((arange(0, hmax), arange(hmin, -1)))
    w = np.concatenate((arange(0, hmax), arange(hmin, -1)))
    h2, w2 = np.meshgrid(h, w)
    magnitude = np.sqrt(np.power(h2, 2.0) + np.power(w2, 2.0))
    angle = np.arctan2(h2, w2)
    
    magnitude_filter = point_operator(magnitude, xm, ym)
    angle_filter = point_operator(angle, xa, ya)
    bandpass_filter = magnitude_filter * angle_filter
    
    return bandpass_filter

def normalize_video(input_video, v_min=None, v_max=None):
    video = input_video
    if v_min is not None:
        video[video < v_min] = v_min
    video = video - np.amin(video)
    if v_max is not None:
        v_max = v_max - v_min
        video[v_max < video] = v_max
    video = video / np.amax(video)
    video = video * 255.9999
    output_video = video.astype('uint8')
    return output_video

def downsample_video(input_dft_video):
    """
    Downsample a video (expressed in the Fourier domain)
    """
    
    n, h, w = input_dft_video.shape
    if h % 4 == 0:
        hmin = 1 * h // 4
        hmax = 3 * h // 4
    else:
        raise NotImplementedError()
    if w % 4 == 0:
        wmin = 1 * w // 4
        wmax = 3 * w // 4
    else:
        raise NotImplementedError()
    
    # Downsample frame by frame
    output_dft_video = np.empty((n, h // 2, w // 2), dtype='complex')
    for index, input_dft_frame in enumerate(input_dft_video):
        output_dft_frame = input_dft_frame[hmin:hmax, wmin:wmax]
        output_dft_video[index] = output_dft_frame
    
    return output_dft_video

def upsample_image(input_dft_image, times=1):
    """
    Upsample an image (represented in the Fourier domain)
    """
    
    if times == 0:
        
        return input_dft_image
    
    else:
        
        h, w = input_dft_image.shape
        hmin = 1 * h // 2
        hmax = 3 * h // 2
        wmin = 1 * w // 2
        wmax = 3 * w // 2
        
        # Upsample frame
        output_dft_image = np.zeros((2 * h, 2 * w))
        output_dft_image[hmin:hmax, wmin:wmax] = np.fft.fftshift(input_dft_image)
        output_dft_image[hmax, wmin:wmax] = output_dft_image[hmin, wmin:wmax]
        output_dft_image[hmin:hmax, wmax] = output_dft_image[hmin:hmax, wmin]
        output_dft_image = np.fft.ifftshift(output_dft_image)
        
        return upsample_image(output_dft_image, times=times-1)

def compute_steerable_pyramid(input_video, n_levels, n_directions, n_octaves):
    
    dft_video = np.empty(input_video.shape, dtype='complex')
    for index, frame in enumerate(input_video):
        dft_frame = np.fft.fft2(frame)
        dft_video[index] = dft_frame
    
    shape = dft_video.shape
    height = shape[1]
    width = shape[2]
    
    n_bands_per_octave = np.around(1.0 / n_octaves)
    
    n_videos = 1 + n_directions * (n_levels - 2) + 1
    output_magnitude_videos = [None] * n_videos
    output_phase_videos = [None] * n_videos
    
    plt.figure()
    n_rows = int(np.ceil(np.sqrt(n_videos)))
    n_cols = int(np.ceil(n_videos / n_rows))
    
    # Compute highpass video
    level = 0
    highpass_magnitude_video = np.empty(shape)
    highpass_phase_video = np.empty(shape)
    highpass_filter = design_highpass_filter(height, width, level, n_octaves)
    #>>>>
    ax0 = plt.subplot(n_rows, n_cols, 1)
    ax0.set_adjustable('box-forced')
    image = np.fft.fftshift(highpass_filter)
    plt.imshow(image)
    plt.colorbar()
    plt.axis('off')
    sum_filter = highpass_filter
    #<<<<
    for index, dft_frame in enumerate(dft_video):
        highpass_dft_frame = dft_frame * highpass_filter
        highpass_frame = np.fft.ifft2(highpass_dft_frame)
        highpass_magnitude_frame = np.abs(highpass_frame)
        highpass_magnitude_video[index] = highpass_magnitude_frame
        highpass_phase_frame = np.angle(highpass_frame)
        highpass_phase_video[index] = highpass_phase_frame
    output_magnitude_videos[0] = highpass_magnitude_video
    output_phase_videos[0] = highpass_phase_video
    
    # Compute oriented bandpass video
    for level in range(1, n_levels - 1):
        band = level % n_bands_per_octave
        for direction in range(0, n_directions):
            bandpass_magnitude_video = np.empty(shape)
            bandpass_phase_video = np.empty(shape)
            bandpass_filter = design_bandpass_filter(height, width, level, direction, band, n_directions, n_octaves)
            #>>>>
            times = (level - 1) // n_bands_per_octave
            ax = plt.subplot(n_rows, n_cols, 1 + 1 + (level - 1) * n_directions + direction, sharex=ax0, sharey=ax0)
            ax.set_adjustable('box-forced')
            image = upsample_image(bandpass_filter, times=times)
            image = np.fft.fftshift(image)
            plt.imshow(image)
            plt.colorbar()
            plt.axis('off')
            sum_filter = sum_filter + upsample_image(bandpass_filter, times=times)
            #<<<<
            for index, dft_frame in enumerate(dft_video):
                bandpass_dft_frame = dft_frame * bandpass_filter
                bandpass_frame = np.fft.ifft2(bandpass_dft_frame)
                bandpass_magnitude_frame = np.abs(bandpass_frame)
                bandpass_magnitude_video[index] = bandpass_magnitude_frame
                bandpass_phase_frame = np.angle(bandpass_frame)
                bandpass_phase_video[index] = bandpass_phase_frame
            output_index = 1 + (level - 1) * n_directions + direction
            #>>>>
            print("Output index:")
            print("  {}".format(output_index))
            #<<<<
            output_magnitude_videos[output_index] = bandpass_magnitude_video
            output_phase_videos[output_index] = bandpass_phase_video
        if band == 0:
            # Downsampling
            # >>>>>
            print("Downsampling...")
            #<<<<
            dft_video = downsample_video(dft_video)
            shape = dft_video.shape
            height = shape[1]
            width = shape[2]
        else:
            pass
    
    # Compute lowpass video
    level = n_levels - 1
    band = level % n_bands_per_octave
    lowpass_magnitude_video = np.empty(shape)
    lowpass_phase_video = np.empty(shape)
    lowpass_filter = design_lowpass_filter(height, width, level, band, n_octaves)
    #>>>>
    times = (level - 1) // n_bands_per_octave
    ax = plt.subplot(n_rows, n_cols, 1 + 1 + (n_levels - 2) * n_directions, sharex=ax0, sharey=ax0)
    ax.set_adjustable('box-forced')
    image = upsample_image(lowpass_filter, times=times)
    image = np.fft.fftshift(image)
    plt.imshow(image)
    plt.colorbar()
    plt.axis('off')
    sum_filter = sum_filter + upsample_image(lowpass_filter, times=times)
    #<<<<
    #>>>>
    ax = plt.subplot(n_rows, n_cols, 1 + 1 + (n_levels - 2) * n_directions + 1, sharex=ax0, sharey=ax0)
    ax.set_adjustable('box-forced')
    image = np.fft.fftshift(sum_filter)
    plt.imshow(image)
    plt.colorbar()
    plt.axis('off')
    #<<<<
    for index, dft_frame in enumerate(dft_video):
        lowpass_dft_frame = dft_frame * lowpass_filter
        lowpass_frame = np.fft.ifft2(lowpass_dft_frame)
        lowpass_magnitude_frame = np.abs(lowpass_frame)
        lowpass_magnitude_video[index] = lowpass_magnitude_frame
        lowpass_phase_frame = np.angle(lowpass_frame)
        lowpass_phase_video[index] = lowpass_phase_frame
    output_magnitude_videos[-1] = lowpass_magnitude_video
    output_phase_videos[-1] = lowpass_phase_video
    
    plt.show()
    
    return output_magnitude_videos, output_phase_videos

def get_path(input_path, addition=None):
    if addition is None:
        addition = "output"
    root_path, extension = os.path.splitext(input_path)
    output_path = "".join([root_path, "_", addition, extension])
    return output_path


def center_phase(input_phase_pyramid, mask_pyramid=None):
    output_phase_pyramid = [None] * len(input_phase_pyramid)
    for index, input_phase_video in enumerate(input_phase_pyramid):
        if mask_pyramid is None:
            phase_video = input_phase_video
        else:
            phase_video = input_phase_video * mask_pyramid[index]
        dft_video = np.exp(1j * phase_video)
        mean_dft_video = np.mean(dft_video, axis=0)
        magnitude_mean_dft_video = np.abs(mean_dft_video)
        mask = 0.0 < magnitude_mean_dft_video
        mean_dft_video[mask] = mean_dft_video[mask] / magnitude_mean_dft_video[mask]
        dft_video = dft_video * np.conjugate(mean_dft_video)
        output_phase_video = np.angle(dft_video)
        output_phase_pyramid[index] = output_phase_video
    return output_phase_pyramid


def get_mask_pyramid(magnitude_pyramid):
    mask_pyramid = [None] * len(magnitude_pyramid)
    for index, magnitude_video in enumerate(magnitude_pyramid):
        max_magnitude_frame = np.amax(magnitude_video, axis=0)
        median_magnitude = np.median(magnitude_video)
        mask_frame = median_magnitude <= max_magnitude_frame
        mask_pyramid[index] = mask_frame
    return mask_pyramid


if __name__ == "__main__":
    
    argv = sys.argv
    input_path = argv[1]
    output_path = argv[2]

    # TODO: remove first attempt
    # analyze_spatial_gaussian_down_temporal_ideal(input_path, output_path)
    
    input_video = load_tiff(input_path)
    # frac = 16
    frac = 1
    input_video = input_video[:input_video.shape[0] // frac, :, :]
    print("Input video shape:")
    print("  {}".format(input_video.shape))
    
    n_octaves = 0.5 # default value (c.f. \cite{wadhwa2013phase})
    # n_octaves = 0.5
    n_levels = 1 + int(3.0 / n_octaves) + 1 # default value (i.e. highpass level + bandpass levels + lowpass level, c.f. \cite{wadhwa2013phase})
    # n_levels = 1 + 3 + 1
    n_directions = 8 # default value (c.f. \cite{wadhwa2013phase})
    # n_directions = 8
    magnitude_pyramid, phase_pyramid = compute_steerable_pyramid(input_video, n_levels, n_directions, n_octaves)
    
    
    # Filter phase outputs
    mask_pyramid = get_mask_pyramid(magnitude_pyramid)
    phase_pyramid = center_phase(phase_pyramid, mask_pyramid=mask_pyramid)
    
    
    # Normalize videos
    for i in range(0, len(magnitude_pyramid)):
        magnitude_pyramid[i] = normalize_video(magnitude_pyramid[i])
    for i in range(0, len(phase_pyramid)):
        phase_pyramid[i] = normalize_video(phase_pyramid[i], v_min=-np.pi, v_max=np.pi)
    
    # Save outputs
    
    level = 0
    index = 0
    addition = "band_level_{}_highpass_magnitude".format(level)
    output_path = get_path(input_path, addition=addition)
    save_tiff(magnitude_pyramid[index], output_path)
    addition = "band_level_{}_highpass_phase".format(level)
    output_path = get_path(input_path, addition=addition)
    save_tiff(phase_pyramid[index], output_path)
    
    for level in range(1, n_levels - 1):
        for direction in range(0, n_directions):
            index = 1 + (level - 1) * n_directions + direction
            addition = "band_level_{}_bandpass_direction_{}_magnitude".format(level, direction)
            output_path = get_path(input_path, addition=addition)
            save_tiff(magnitude_pyramid[index], output_path)
            addition = "band_level_{}_bandpass_direction_{}_phase".format(level, direction)
            output_path = get_path(input_path, addition=addition)
            save_tiff(phase_pyramid[index], output_path)
    
    level = n_levels - 1
    index = 1 + (level - 1) * n_directions
    addition = "band_level_{}_lowpass_magnitude".format(level)
    output_path = get_path(input_path, addition=addition)
    save_tiff(magnitude_pyramid[index], output_path)
    addition = "band_level_{}_lowpass_phase".format(level)
    output_path = get_path(input_path, addition=addition)
    save_tiff(phase_pyramid[index], output_path)
