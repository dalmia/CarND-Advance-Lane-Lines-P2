import numpy as np
import cv2
import glob

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    return grad_binary

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    mag_sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    scaled_sobel = np.uint8(255 * mag_sobel / np.max(mag_sobel))
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    
    return mag_binary

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    dir_binary = np.zeros_like(grad_dir)
    dir_binary[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
    return dir_binary

def hls_channel_threshold(img, channel, thresh=50):
    # Threshold based on H, L or S channel values
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    index_map = {
        'h': 0,
        'l': 1,
        's': 2
    }
    index = index_map[channel]
    channel = hls[:, :, index]
    hls_binary = np.zeros_like(channel)
    hls_binary[channel >= thresh] = 1
    return hls_binary

def lab_channel_threshold(img, channel, thresh=50):
    # Threshold based on L, A or B channel values
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    index_map = {
        'l': 0,
        'a': 1,
        'b': 2
    }
    index = index_map[channel]
    channel = lab[:, :, index]
    lab_binary = np.zeros_like(channel)
    lab_binary[channel >= thresh] = 1
    return lab_binary

def r_channel_threshold(img, thresh=200):
    channel_r = img[:, :, 0]
    r_binary = np.zeros_like(channel_r)
    r_binary[channel_r >= thresh] = 1
    return r_binary