import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL as pil
import PIL.Image
import tifffile as tf



def change_extension(input_path, extension):
    name = input_path.split(".")[0]
    output_path = "{}.{}".format(name, extension)
    return output_path


def save_tiff(frames, output_path):
    frames = frames.astype('uint8')
    tf.imsave(output_path, frames, imagej=True)
    return


def load_tiff(path):
    # Open the TIFF file with PIL.
    video = pil.Image.open(path)
    # # Examine the file contents.
    # print('  format: {}'.format(video.format))
    # print('   width: {}'.format(video.width))
    # print('  height: {}'.format(video.height))
    # print('    mode: {}'.format(video.mode))
    # print('n frames: {}'.format(video.n_frames))
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
        frames[index, :, :] = frame
    # Return the frames.
    return frames


def save_avi(frames, path, fourcc="MJPG", fps=50.0, isColor=True):
    fourcc = cv2.VideoWriter_fourcc(*fourcc)
    width = frames.shape[2]
    height = frames.shape[1]
    size = (width, height)
    video = cv2.VideoWriter(path, fourcc, fps, size, isColor=isColor)
    n_frames = frames.shape[0]
    for index in range(n_frames):
        # TODO: add a progress bar...
        frame = frames[index, :, :]
        frame = np.dstack((frame, frame, frame)) # recompose the three RGB layer
        video.write(frame)
    video.release()
    return


def load_avi(path, verbose=False):
    video = cv2.VideoCapture(path)
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    shape = (n_frames, height, width)
    dtype = 'uint8'
    if verbose:
        print("  number of frames: {}".format(n_frames))
        print("  height          : {}".format(height))
        print("  width           : {}".format(width))
        print("  dtype           : {}".format(dtype))
    frames = np.empty(shape, dtype=dtype)
    for index in range(n_frames):
        frame = video.read()[1]
        frame = frame[:, :, 0] # keep only one RGB layer
        frames[index, :, :] = frame
    return frames


def crop(frames, offsets):
    left_offset = offsets[0]
    right_offset = offsets[1]
    up_offset = offsets[2]
    down_offset = offsets[3]
    height = frames.shape[1]
    width = frames.shape[2]
    frames = frames[:, up_offset:height-down_offset, left_offset:width-right_offset]
    return frames


def histogram(frames):
    data = frames.flatten()
    hist, _ = np.histogram(data, bins=256, range=(0.0, 256.0))
    return hist


def cumulative_histogram(frames):
    hist = histogram(frames)
    cumhist = np.cumsum(hist)
    cumhist = cumhist / np.amax(cumhist)
    return cumhist


def histograms(frames):
    n_frames = frames.shape[0]
    shape = (256, n_frames)
    hists = np.empty(shape)
    for index in range(0, n_frames):
        frame = frames[index, :, :]
        data = frame.flatten()
        hist, _ = np.histogram(data, bins=256, range=(0.0, 256.0))
        hists[:, index] = hist
    return hists


def cumulative_histograms(frames):
    hists = histograms(frames)
    cumhists = np.cumsum(hists, axis=0)
    cumhists = cumhists / np.amax(cumhists)
    return cumhists


index = 0

def show_histograms(frames):
    n_frames = frames.shape[0]
    hist = cumulative_histogram(frames)
    hists = cumulative_histograms(frames)
    def key_press_event(event):
        global index
        key = event.key
        if key == 'left':
            index = (index - 1) % (n_frames + 1)
        elif key == 'right':
            index = (index + 1) % (n_frames + 1)
        elif key == 'escape':
            pass
        else:
            pass
        ax.set_title('frame {}/{}'.format(index, n_frames))
        if index == 0:
            for rect, height in zip(rects, hist):
                rect.set_height(height)
        else:
            for rect, height in zip(rects, hists[:, index-1]):
                rect.set_height(height)
        fig.canvas.draw()
        return
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, 256)
    ax.set_ylim(0, np.amax(hists))
    ax.set_title('frame {}/{}'.format(index, n_frames))
    fig.canvas.mpl_connect('key_press_event', key_press_event)
    rects = ax.bar(np.linspace(0.0, 255.0, 256), hist, width=1.0, color='green')
    plt.show()
    return


def match_cumulative_histograms(frames):
    cum_hist = cumulative_histogram(frames)
    cum_hists = cumulative_histograms(frames)
    n_frames = frames.shape[0]
    matches = np.zeros(n_frames)
    for index in range(0, n_frames):
        offset_min = -10
        offset_max = +10
        shape = offset_max - offset_min + 1
        errors = np.empty(shape)
        offsets = range(offset_min, offset_max + 1)
        for offset in offsets:
            tmp = np.zeros(256)
            for i in range(0, 256):
                tmp[i] = cum_hists[max(0, min(i + offset, 255)), index]
            errors[offset - offset_min] = np.sum(np.abs(tmp - cum_hist))
        i = np.argmin(errors)
        matches[index] = offsets[i]
    for index in range(0, n_frames):
        frame = frames[index, :, :]
        frame = frame + offset
        if offset < 0:
            frame[255 + offset < frame] = 0
        elif 0 < offset:
            frame[frame < offset] = 255
        frames[index, :, :] = frame
    return frames
