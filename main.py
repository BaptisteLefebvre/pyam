import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import scipy as sp
import scipy.signal
import sys


image_path = 'testmovie kept stack aligned.tif'
offset_left = 600 # offset (in pixels) to remove the left part of the frame
offset_right = 200 # ...
offset_up = 50 # ...
offset_down = 50 # ...
#n_frames = 100
n_frames = None


def display_avi(path):
    cam = cv2.VideoCapture(path)
    n_frames = int(cam.get(7))
    for index in range(n_frames):
        img = cam.read()[1]
        cv2.imshow(path, img)
        if cv2.waitKey(1) == 27: # ESC key
            break
    cv2.waitKey()
    return

def save_avi(frames, path):
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    fps = 20.0
    n_frames = frames.shape[0]
    width = frames.shape[2]
    height = frames.shape[1]
    size = (width, height)
    video = cv2.VideoWriter(path, fourcc, fps, size, isColor=True)
    for index in range(n_frames):
        frame = frames[index, :, :]
        frame = np.dstack((frame, frame, frame)) # recompose the three RGB layer
        video.write(frame)
    cv2.waitKey()
    video.release()
    return

# Open the TIFF file with PIL
image = Image.open(image_path)

# Examine the file contents
print('  format: {}'.format(image.format))
print('   width: {}'.format(image.width))
print('  height: {}'.format(image.height))
print('    mode: {}'.format(image.mode))
print('n frames: {}'.format(image.n_frames))

# Retrieve the different frames
data = []
if n_frames is None:
    n_frames = image.n_frames
for frame in range(n_frames):
    image.seek(frame)
    data.append(np.array(image))
data = np.array(data)

# Keep the region of interest for each video frame
shape = data.shape
data = data[:, offset_up:shape[1]-offset_down, offset_left:shape[2]-offset_right]

# Keep only the frames of interest
data = data[:n_frames, :, :]

# Save the AVI file with OpenCV
video_path = 'testmovie kept stack aligned.avi'
force = False
if not os.path.isfile(video_path) or force:
    save_avi(data, video_path)

# # Read the AVI file with OpenCV
# display_avi(video_path)
# cv2.destroyAllWindows()

# # Read from camera with OpenCV
# display_avi(0)
# cv2.destroyAllWindows()

# Luminance correction
for index in range(n_frames):
    tmp = data[index, :, :]
    lum = np.mean(tmp).astype('uint8')
    if lum < 128:
        tmp[255 - (128 - lum) < tmp] = 255 - (128 - lum)
        tmp = tmp + (128 - lum)
    if 128 < lum:
        tmp[tmp < lum - 128] = lum - 128
        tmp = tmp - (lum - 128)
    data[index, :, :] = tmp


# Create the AVI with background substraction
median = np.median(data, axis=0)
amin = np.amin(data[0, :, :].astype('float') - median)
for index in range(n_frames):
    bmin = np.amin(data[index, :, :].astype('float') - median)
    if bmin < amin:
        amin = bmin
amax = np.amax(data[0, :, :].astype('float') - median) - amin
for index in range(n_frames):
    bmax = np.amax(data[index, :, :].astype('float') - median) - amin
    if amax < bmax:
        amax = bmax
for index in range(n_frames):
    print('Processing frame {}...'.format(index))
    tmp = data[index, :, :].astype('float')
    tmp = tmp - median
    tmp = tmp - amin
    tmp = tmp / amax
    #####
    kernel = (1.0 / 16.0) * np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]])
    tmp = sp.signal.convolve2d(tmp, kernel, mode='same')
    lum = np.median(tmp)
    tmp = tmp - (lum - 0.5)
    #####
    #####
    # tmp = 255.999 * tmp
    # or
    #tmin = 0.66
    #tmax = 0.78
    tmin = 0.45
    tmax = 0.55
    tmp[tmp < tmin] = tmin
    tmp[tmax < tmp] = tmax
    tmp = tmp - tmin
    tmp = tmp / (tmax - tmin)
    tmp = 255.999 * tmp
    #####
    tmp = tmp.astype('uint8')
    data[index, :, :] = tmp

# Save the AVI file with OpenCV
output_path = 'output.avi'
force = True
if not os.path.isfile(output_path) or force:
    save_avi(data, output_path)

# # Display the AVI with background substraction
# display_avi(bg_sub_path)


lum = np.mean(np.mean(data, axis=2), axis=1)
plt.figure()
plt.subplot(1, 1, 1)
plt.xlabel('frame')
plt.ylabel('luminance')
plt.xlim(0, n_frames - 1)
plt.ylim(0, 255)
plt.grid()
plt.plot(lum)
plt.show()
