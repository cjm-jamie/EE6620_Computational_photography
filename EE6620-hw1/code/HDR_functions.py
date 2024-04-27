''' Functions in HDR flow '''

import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

Z_max = 255
Z_min = 0
gamma = 2.2

def ReadImg(path, flag=1):
    img = cv.imread(path, flag)  # flag = 1 means to load a color image
    img = img[:,:,[2,1,0]]
    return img


def SaveImg(img, path):
    img = img[:,:,[2,1,0]]
    cv.imwrite(path, img)
    

def LoadExposures(source_dir):
    """ load bracketing images folder

    Args:
        source_dir (string): folder path containing bracketing images and a image_list.txt file
                             image_list.txt contains lines of image_file_name, exposure time, ... 
    Returns:
        img_list (uint8 ndarray, shape (N, height, width, ch)): N bracketing images (3 channel)
        exposure_times (list of float, size N): N exposure times
    """
    filenames = []
    exposure_times = []
    f = open(os.path.join(source_dir, 'image_list.txt'))
    for line in f:
        if (line[0] == '#'):
            continue
        (filename, exposure, *_) = line.split()
        filenames += [filename]
        exposure_times += [float(exposure)]
    img_list = [ReadImg(os.path.join(source_dir, f)) for f in filenames]
    img_list = np.array(img_list)
    
    return img_list, exposure_times


def PixelSample(img_list):
    """ Sampling

    Args:
        img_list (uint8 ndarray, shape (N, height, width, ch))
        
    Returns:
        sample (uint8 ndarray, shape (N, height_sample_size, width_sample_size, ch))
    """
    # trivial periodic sample
    # sample every 64th pixel along height and weight
    sample = img_list[:, ::64, ::64, :]
    
    return sample


## custom function to get weight according to the pixel value
def GetWeight(pixel_value):
    """

    Args:
        pixel_value(np.uint8)
    
    Returns:
        weight (np.float64): The weight corresponding to the input pixel intensity   
    """
    if pixel_value <= (Z_min + Z_max)/2:
        return pixel_value - Z_min
    return Z_max - pixel_value


## custom function to random sample pixels for plotting result
def random_sample_pixels(img_samples, num_samples=3):
    """
    Randomly sample specified number of non-repeating pixel positions from img_samples.
    Args:
    - img_samples: Input image samples, with shape (N, height_sample_size, width_sample_size)
    - num_samples: Number of pixels to sample, defaults to 3
    Returns:
    - sampled_pixels: List of sampled pixel positions, each pixel position represented as (i, j)
    """
    height, width = img_samples.shape[1], img_samples.shape[2]

    all_pixels = np.array([(i, j) for i in range(height) for j in range(width)])

    sampled_indices = np.random.choice(len(all_pixels), size=num_samples, replace=False)
    sampled_pixels = all_pixels[sampled_indices]

    return sampled_pixels


def EstimateResponse(img_samples, etime_list, lambda_=50):
    """ Estimate camera response for bracketing images

    Args:
        img_samples (uint8 ndarray, shape (N, height_sample_size, width_sample_size)): N bracketing sampled images (1 channel)
        etime_list (list of float, size N): N exposure times
        lambda_ (float): Lagrange multiplier (Defaults to 50)
    
    Returns:
        response (float ndarray, shape (256)): response map
    """
    reshape_img_samples = img_samples.reshape(img_samples.shape[0], -1)
    num_sample_imgs = reshape_img_samples.shape[0]
    num_sample_pixels = reshape_img_samples.shape[1]

    # M(# of equations) = # of sample imgs * # of sample pixels + 254(smoothness term) + 1
    # N(# of variables) = 256 + # of sample pixels
    # mat_A: M(# of equations) * N(# of variables)
    # mat_a = np.zeros((num_sample_imgs * num_sample_pixels + 1, num_sample_pixels + 256), dtype=np.float64)
    mat_a = np.zeros((num_sample_imgs * num_sample_pixels + 1 + 254, num_sample_pixels + 256), dtype=np.float64)
    mat_b = np.zeros((mat_a.shape[0], 1), dtype=np.float64)

    # 1. data term assignment
    # k: k-th measurement, zij: sample pixel value from reshape_sample_imgs
    k = 0
    for i in range(num_sample_pixels):
        for j in range(num_sample_imgs):
            zij = reshape_img_samples[j,i]
            wij = GetWeight(zij)
            mat_a[k,zij] = wij
            mat_a[k,256+i] = -wij
            mat_b[k, 0] = wij * np.log(etime_list[j])
            k+=1

    # 2. smoothness term assignment, add another 264 smoothness term function
    for zk in range(Z_min+1, Z_max):
        wij = GetWeight(zk)
        mat_a[k, zk-1] = lambda_ * wij
        mat_a[k, zk] = -2 * lambda_ * wij
        mat_a[k, zk+1] = lambda_ * wij
        k+=1

    # 3. add unit exposure assumption by letting g(127)=0
    mat_a[k, 127] = 1

    # get x in Ax=b by minimize square error
    x, residuals, rank, s = np.linalg.lstsq(mat_a, mat_b)

    # only take first 256 row as output
    # response[0] means g[0]
    response = x[:256,0]

    # sampled pixel(default=3) to plot the response curve
    sampled_pixels_coord = random_sample_pixels(img_samples)

    # Plot and save figure 1: logt vs. sample pixel value
    plt.figure(figsize=(8, 6))
    plt.xlabel('log(exposure_time)')
    plt.ylabel('Pixel Value')
    plt.title('Pixel Value vs. log(exposure_time)')
    color = ['red', 'green', 'blue']
    for sample_pixels in range(3):
        pixel_values = img_samples[:,sampled_pixels_coord[sample_pixels][0], sampled_pixels_coord[sample_pixels][1]]
        plt.plot(np.log(etime_list), img_samples[:,sampled_pixels_coord[sample_pixels][0], sampled_pixels_coord[sample_pixels][1]], marker='o', color=color[sample_pixels], alpha=0.5)
    plt.grid(True)
    # plt.savefig('../Result/research_study/wo_smooth/logt_vs_pixel_value.png')
    plt.savefig('../Result/logt_vs_pixel_value.png')

    # Plot and save figure 2: radiance vs. sample pixel value
    plt.figure(figsize=(8, 6))
    for sample_pixels in range(3):
        pixel_values = img_samples[:,sampled_pixels_coord[sample_pixels][0], sampled_pixels_coord[sample_pixels][1]]
        plt.plot(response[pixel_values], pixel_values, marker='o', color=color[sample_pixels], alpha=0.5)
    plt.xlabel('Radiance')
    plt.ylabel('Pixel Value')
    plt.title('Pixel Value vs. Radiance')
    plt.grid(True)
    # plt.savefig('../Result/research_study/wo_smooth/radiance_vs_pixel_value.png')
    plt.savefig('../Result/radiance_vs_pixel_value.png')

    # Plot and save figure 3: radiance vs. all pixel value
    plt.figure(figsize=(8, 6))
    all_pixel_values = np.arange(len(response))
    plt.plot(response[all_pixel_values], all_pixel_values, marker='o', color=color[sample_pixels], alpha=0.5)
    plt.xlabel('Radiance')
    plt.ylabel('Pixel Value')
    plt.title('Pixel Value vs. Radiance')
    plt.grid(True)
    # plt.savefig('../Result/research_study/wo_smooth/radiance_vs_all_pixel_value.png')
    plt.savefig('../Result/radiance_vs_all_pixel_value.png')

    return response


def ConstructRadiance(img_list, response, etime_list):
    """ Construct radiance map from brackting images

    Args:
        img_list (uint8 ndarray, shape (N, height, width)): N bracketing images (1 channel)
        response (float ndarray, shape (256)): response map
        etime_list (list of float, size N): N exposure times
    
    Returns:
        radiance (float ndarray, shape (height, width)): radiance map
    """
    # img_pixel: # of pixel per image
    img_pixel = img_list[0].shape
    radiance = np.zeros(img_pixel, dtype=np.float64)

    # num_imgs: # of images j
    num_imgs = img_list.shape[0]

    # final radiance for each pixel i can be estimated by averaging measurements from all images j
    for i in range(img_pixel[0]):
        for j in range(img_pixel[1]):
            # iterate over images to compute the average exposure of each pixel
            g = []
            wij = []
            for k in range(num_imgs):
                # response of each pixel in different images
                g.append(response[img_list[k][i, j]])
                wij.append(GetWeight(img_list[k][i, j]))
            g = np.array(g)
            wij = np.array(wij)
            sum_wij = np.sum(wij)
            if sum_wij > 0:
                log_radiance = np.sum(wij * (g - np.log(etime_list))) / np.sum(wij)
            else:
                log_radiance = g[num_imgs // 2] - np.log(etime_list[num_imgs // 2])
            radiance[i, j] = np.exp(log_radiance)

    return radiance


def CameraResponseCalibration(src_path, lambda_):
    # img_list.shape = (N, height, width, ch)
    # radiance.shape = (height, width, ch)
    img_list, exposure_times = LoadExposures(src_path)
    radiance = np.zeros_like(img_list[0], dtype=np.float32)
    pixel_samples = PixelSample(img_list)
    # compute radiance for rgb channel respectively
    for ch in range(3):
        response = EstimateResponse(pixel_samples[:, :,:, ch], exposure_times, lambda_)
        radiance[:,:,ch] = ConstructRadiance(img_list[:,:,:,ch], response, exposure_times)

    return radiance


def WhiteBalance(src, y_range, x_range, case = 'red'):
    """ White balance based on Known to be White(KTBW) region

    Args:
        src (float ndarray, shape (height, width, ch)): source radiance
        y_range (tuple of 2 int): location range in y-dimension
        x_range (tuple of 2 int): location range in x-dimension
        
    Returns:
        result (float ndarray, shape (height, width, ch))
    """
   
    # Extract the region of interest
    # why region[0][0] = array([673.99243, 537.7938 , 308.81683], dtype=float32) ??? > 256??
    region = src[y_range[0]:y_range[1], x_range[0]:x_range[1]]
    
    # Calculate the average radiance in the "known to be white" region for each color channel
    Ravg = np.mean(region[:,:,0])
    Gavg = np.mean(region[:,:,1])
    Bavg = np.mean(region[:,:,2])
    
    if case == 'red':
        # Scale the G and B channels using the ratios of Ravg / Gavg, Ravg / Bavg, respectively
        result = src.copy()
        result[:,:,1] *= Ravg / Gavg
        result[:,:,2] *= Ravg / Bavg

    if case == 'green':
        # Scale the R and B channels using the ratios of Gavg / Ravg, Gavg / Bavg, respectively
        result = src.copy()
        result[:,:,0] *= Gavg / Ravg
        result[:,:,2] *= Gavg / Bavg

    if case == 'blue':
        # Scale the R and G channels using the ratios of Bavg / Ravg, Bavg / Gavg, respectively
        result = src.copy()
        result[:,:,0] *= Bavg / Ravg
        result[:,:,1] *= Bavg / Gavg

    return result

def GlobalTM(src, scale=1.0, case='default'):
    """ Global tone mapping

    Args:
        src (float ndarray, shape (height, width, ch)): source radiance image
        scale (float): scaling factor (Defaults to 1.0)
    
    Returns:
        result(uint8 ndarray, shape (height, width, ch)): result HDR image
    """

    result = np.zeros_like(src, dtype=np.float32)

    # compressed RGB to do global tone mapping
    for ch in range(3):
        x_max = np.max(src[:,:,ch])
        log2_x = scale * (np.log2(src[:,:,ch])-np.log2(x_max)) + np.log2(x_max)
        x_hat = np.exp2(log2_x)
        # apply gamma correction
        x_prime = np.power(x_hat, 1/gamma)
        if case == 'default':
            result[:,:,ch] = np.round((np.clip(x_prime, 0, 1) * 255))
        elif case == 'his_stretch':
            # Histogram stretching
            for ch in range(3):
                min_value = np.min(src[:,:,ch])
                max_value = np.max(src[:,:,ch])
                stretched_channel = ((src[:,:,ch] - min_value) / (max_value - min_value)) * 255
                result[:,:,ch] = np.round(np.clip(stretched_channel, 0, 255))
        elif case == 'log_scaling':
            # Logarithmic scaling
            for ch in range(3):
                # Add a small epsilon to avoid taking the logarithm of zero
                epsilon = 1e-6
                log_scaled_channel = np.log(src[:,:,ch] + epsilon)
                # Normalize to cover the full dynamic range (0 to 255)
                log_scaled_channel = (log_scaled_channel - np.min(log_scaled_channel)) / (np.max(log_scaled_channel) - np.min(log_scaled_channel))
                result[:,:,ch] = np.round(log_scaled_channel * 255)
    # plt.clf()
    # # for research purposes, plot the histogram 
    # histogram, bins = np.histogram(result.flatten(), bins=256, range=(0, 255))
    # plt.plot(bins[:-1], histogram, color='r', label=case)
    # plt.xlabel('Pixel Value')
    # plt.ylabel('Frequency')
    # plt.title(f'Histogram of {case} gamma correction') 
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(f'../Result/research_study/clip/{case}.png')

    return result


def LocalTM(src, imgFilter, scale=3.0, case = 'default'):
    """ Local tone mapping

    Args:
        src (float ndarray, shape (height, width,ch)): source radiance image
        imgFilter (function): filter function with preset parameters
        scale (float): scaling factor (Defaults to 3.0)
    
    Returns:
        result(uint8 ndarray, shape (height, width,ch)): result HDR image
    """
    
    # Separate Intensity map (I) and Color Ratios (Cr, Cg, Cb) for radiance (R, G, B)
    Cx = np.zeros_like(src, dtype=np.float32)
    I = np.mean(src, axis=2)
    for ch in range(3):
        Cx[:,:,ch] = src[:,:,ch]/I

    # Take the logarithm of the intensity map
    L = np.log2(I)

    # Separate the detail layer (LD) and base layer (LB) of the logarithm of intensity with a Gaussian filter
    LB = imgFilter(L)
    LD = L - LB

    # Compress the contrast of the base layer
    Lmin = np.min(LB)
    Lmax = np.max(LB)
    L_B_prime = (LB - Lmax) * scale / (Lmax - Lmin)

    # Reconstruct the intensity map with the adjusted base layer and detail layer
    I_prime = np.power(2, (L_B_prime + LD))

    # Reconstruct the color map with the adjusted intensity and color ratio
    result = np.zeros_like(src)
    for ch in range(3):
        result[:,:,ch] = Cx[:,:,ch] * I_prime

        # apply gamma correction
        result[:,:,ch] = np.power(result[:,:,ch], 1/gamma)
        if case == 'default':
            result[:,:,ch] = np.round((np.clip(result[:,:,ch], 0, 1) * 255))
        elif case == 'his_stretch':
            # Histogram stretching
            for ch in range(3):
                min_value = np.min(src[:,:,ch])
                max_value = np.max(src[:,:,ch])
                stretched_channel = ((src[:,:,ch] - min_value) / (max_value - min_value)) * 255
                result[:,:,ch] = np.round(np.clip(stretched_channel, 0, 255))
        elif case == 'log_scaling':
            # Logarithmic scaling
            for ch in range(3):
                # Add a small epsilon to avoid taking the logarithm of zero
                epsilon = 1e-6
                log_scaled_channel = np.log(src[:,:,ch] + epsilon)
                # Normalize to cover the full dynamic range (0 to 255)
                log_scaled_channel = (log_scaled_channel - np.min(log_scaled_channel)) / (np.max(log_scaled_channel) - np.min(log_scaled_channel))
                result[:,:,ch] = np.round(log_scaled_channel * 255)

    return result

# for research study, plot the histogram of the HDR image after white balancing and global/local tone mapping
def CompareHistograms(src, y_range, x_range, GaussianFilter, scale=7):
    # List to store histogram data
    histograms = []

    # Loop over different fixed color channels during white balancing
    for case in ['red', 'green', 'blue']:
        # Apply white balancing
        white_balanced_image = WhiteBalance(src, y_range, x_range, case)
        # Apply global/local tone mapping
        # result = GlobalTM(white_balanced_image)
        result = LocalTM(white_balanced_image, GaussianFilter, scale)
        # Calculate histogram of the resulting HDR image
        histogram, bins = np.histogram(result.flatten(), bins=256, range=(0, 255))
        # import pdb
        # pdb.set_trace()
        histograms.append((histogram, bins))

    # Plot histograms
    colors = ['r', 'g', 'b']
    labels = ['Fix R Channel', 'Fix G Channel', 'Fix B Channel']

    histogram, bins = histograms[2]
    color = colors[2]
    label = labels[2]
    plt.clf()
    plt.plot(bins[:-1], histogram, color=color, label=label)

    # for (histogram, bins), color, label in zip(histograms[0], colors[0], labels[0]):
    #     plt.plot(bins[:-1], histogram, color=color, label=label)

    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Comparison of Histograms after White Balancing and Local Tone Mapping')
    plt.legend()
    plt.grid(True)
    plt.savefig('../Result/research_study/white_balance_compare/channelb_lt_white_balance_compare.png')

def GaussianFilter(src, N=35, sigma_s=100):
    """ Gaussian filter

    Args:
        src (float ndarray, shape (height, width)): source intensity
        N (int): window size of the filter (Defaults to 35)
                 filter indices span [-N/2, N/2]
        sigma_s (float): standard deviation of Gaussian filter (Defaults to 100)
    
    Returns:
        result (float ndarray, shape (height, width))
    """

    height, width = src.shape
    result = np.zeros_like(src)

    # Pad the source image to handle borders
    padded_src = np.pad(src, pad_width=N//2, mode='symmetric')

    # Gaussian kernel
    # kernel = np.zeros((N, N))

    # Gaussian kernel
    k, l = np.mgrid[ -N//2 + 1 : N//2 + 1, -N//2 + 1 : N//2 + 1]
    kernel = np.exp(-(k**2 + l**2) / (2 * sigma_s**2))

    # for k in range(-N//2+1, N//2 + 1):
    #     for l in range(-N//2+1, N//2 + 1):
    #         kernel[k + N//2, l + N//2] = np.exp(- (k**2 + l**2) / (2 * sigma_s**2))

    # Normalize the kernel
    kernel /= np.sum(kernel)

    # Apply the filter
    for i in range(height):
        for j in range(width):
            # Extract the kernel region of padded image
            kernel_src = padded_src[i:i+N, j:j+N]
            # Apply the kernel
            result[i, j] = np.sum(kernel_src * kernel)

    return result


def BilateralFilter(src, N=35, sigma_s=100, sigma_r=0.8):
    """ Bilateral filter

    Args:
        src (float ndarray, shape (height, width)): source intensity
        N (int): window size of the filter (Defaults to 35)
                 filter indices span [-N/2, N/2]
        sigma_s (float): spatial standard deviation of bilateral filter (Defaults to 100)
        sigma_r (float): range standard deviation of bilateral filter (Defaults to 0.8)
    
    Returns:
        result (float ndarray, shape (height, width))
    """
    height, width = src.shape
    result = np.zeros_like(src)

    # Pad the source image to handle borders
    padded_src = np.pad(src, pad_width=N//2, mode='symmetric')

    # distance term in bilateral filter (indep. of pixel value)
    k, l = np.mgrid[ -N//2 + 1 : N//2 + 1, -N//2 + 1 : N//2 + 1]
    distance_terms = -(k**2 + l**2) / (2 * sigma_s**2)

    # Bilateral filter
    for i in range(height):
        for j in range(width):
            # Extract the padding kernal
            kernel_src = padded_src[i:i+N, j:j+N]
            pixel_value_terms = -((kernel_src - src[i, j])**2) / (2 * sigma_r**2)
            kernel = np.exp(distance_terms + pixel_value_terms)
            # Normalize the kernel
            kernel /= np.sum(kernel)
            # Apply the kernel
            result[i, j] = np.sum(kernel_src * kernel)

    return result
