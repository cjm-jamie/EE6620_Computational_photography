''' Test functions in deblur flow'''

import time
import numpy as np
import imageio

from PIL import Image
from deconvolution_functions import deconv_Wiener, deconv_RL, deconv_BRL, RL_energy, BRL_energy, kernal_preprocess, BRL_EB
from TV_functions import TVL1, TVL2, TVpoisson

INF = float("inf")
import sys
DBL_MIN = sys.float_info.min

### TEST_PAT_SIZE can be assigned to either 'small' or 'large' ###
#-- 'small' stands for small test pattern size
#-- 'large' stands for large test pattern size
# During implementation, it is recommended to set TEST_PAT_SIZE 'small' for quick debugging.
# However, you have to pass the unit test with TEST_PAT_SIZE 'large' to get the full score in each part.

TEST_PAT_SIZE = 'small'

gamma = 2.2
energy_dict_load = {}

noise_mode = 'pepper'

# Function to add Gaussian noise to an image
def add_gaussian_noise(image, sigma=0.1):
    noise = np.random.normal(scale=sigma, size=image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image

# Function to add pepper noise to an image
def add_pepper_noise(image, amount=0.01):
    noisy_image = np.copy(image)
    height, width = image.shape[:2]
    num_pixels = int(amount * height * width)
    # Set randomly selected pixels to either 0 or 255
    for _ in range(num_pixels):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        noisy_image[y, x] = np.random.choice([0, 255])
    return noisy_image

########################################################################
# here convert image to RGB channel, remove alpha channel
def read_image(file_path):
    img_in = np.asarray(Image.open(file_path))
    img_in = np.transpose(img_in, (2,0,1))

    return img_in

########################################################################
# here convert image to RGB channel, then convert to gray scale
def read_kernel(file_path):
    k_in_RGB = np.asarray(Image.open(file_path).convert('RGB'))
    k_in_gray = np.asarray(Image.fromarray(k_in_RGB).convert('L'))

    return k_in_gray


########################################################################
def write_image(file_path, array):
    array = np.transpose(array, (1, 2, 0))
    imageio.imwrite(file_path , array)
    
########################################################################
def test_Wiener_deconv():
    print ("//--------------------------------------------------------")
    print (f"start Wiener deconvolution, TEST_PAT_SIZE: {TEST_PAT_SIZE}\n")

    # I/O
    img_in = read_image(f'../data/blurred_image_{TEST_PAT_SIZE}/curiosity_medium.png')
    k_in = np.asarray(Image.open('../data/kernel/kernel_medium.png'))
    
    if noise_mode == 'gaussian':
        img_in = add_gaussian_noise(img_in)
    elif noise_mode == 'pepper':
        img_in = add_pepper_noise(img_in)

    # setting 
    SNR_F = 150.0
    to_linear = True
    
    # work
    t_start = time.time()
    k_in = kernal_preprocess(img_in, k_in, to_linear, gamma)
    Wiener_result = deconv_Wiener(img_in, k_in, SNR_F, to_linear, gamma)

    t_end = time.time()
    duration = t_end - t_start

    # print duration
    print(f'    -> processing time = {duration:.2f} sec')

    # store image
    # write_image(f'../MyDeblur_result/result_{TEST_PAT_SIZE}/Wiener_m_SNRF{SNR_F}_rolling_window_{noise_mode}.png' , Wiener_result)
    write_image(f'../result/exploration/{noise_mode}_noise/Wiener_m_SNRF{SNR_F}_rolling_window_{noise_mode}.png' , Wiener_result)

########################################################################
def test_RL():
    print ("//--------------------------------------------------------")
    print (f"start RL deconvolution, TEST_PAT_SIZE: {TEST_PAT_SIZE}\n")
    
    # I/O
    img_in = read_image(f'../data/blurred_image_{TEST_PAT_SIZE}/curiosity_medium.png')
    k_in = np.asarray(Image.open('../data/kernel/kernel_medium.png'))
    if noise_mode == 'gaussian':
        img_in = add_gaussian_noise(img_in)
    elif noise_mode == 'pepper':
        img_in = add_pepper_noise(img_in)

    # setting 
    max_iter_RL = 50
    to_linear = True

    # work
    t_start = time.time()
    RL_result = deconv_RL(img_in, k_in, max_iter_RL, to_linear, gamma)
    t_end = time.time()

    # print duration
    duration = t_end - t_start
    print(f'    -> processing time = {duration:.2f} sec')

    # store image
    # write_image(f'../MyDeblur_result/result_{TEST_PAT_SIZE}/RL_s_iter{max_iter_RL}_{noise_mode}.png' , RL_result)
    write_image(f'../result/exploration/{noise_mode}_noise/RL_s_iter{max_iter_RL}_{noise_mode}.png' , RL_result)

def test_BRL():
    print ("//--------------------------------------------------------")
    print (f"start BRL deconvolution, TEST_PAT_SIZE: {TEST_PAT_SIZE}\n")
    
    # I/O
    img_in = read_image(f'../data/blurred_image_{TEST_PAT_SIZE}/curiosity_medium.png')
    k_in = np.asarray(Image.open('../data/kernel/kernel_medium.png'))
    if noise_mode == 'gaussian':
        img_in = add_gaussian_noise(img_in)
    elif noise_mode == 'pepper':
        img_in = add_pepper_noise(img_in)


    # setting
    max_iter_RL = 50
    rk = 12
    sigma_r = 25.0/255/255
    lamb_da = 0.006/255
    to_linear = True

    # work
    t_start = time.time()
    BRL_result = deconv_BRL(img_in, k_in, max_iter_RL, lamb_da, sigma_r, rk, to_linear, gamma)
    t_end = time.time()

    # print duration
    duration = t_end - t_start
    print(f'    -> processing time = {duration:.2f} sec')

    # store image
    sigma_r = sigma_r * 255 * 255
    lamb_da = lamb_da * 255
    # write_image(f'../MyDeblur_result/result_{TEST_PAT_SIZE}/BRL_s_iter{max_iter_RL}_rk{rk}_si{sigma_r}_lam{lamb_da}_{noise_mode}.png' , BRL_result)
    write_image(f'../result/exploration/{noise_mode}_noise/BRL_s_iter{max_iter_RL}_rk{rk}_si{sigma_r}_lam{lamb_da}_{noise_mode}.png' , BRL_result)

########################################################################
def test_TVL1():    
    print ("//--------------------------------------------------------")
    print (f"start TVL1, TEST_PAT_SIZE: {TEST_PAT_SIZE}\n")
    
    # I/O
    img_in = np.asarray(Image.open(f'../data/blurred_image_{TEST_PAT_SIZE}/curiosity_medium.png'))
    k_in = np.asarray(Image.open('../data/kernel/kernel_medium.png'))  
    if noise_mode == 'gaussian':
        img_in = add_gaussian_noise(img_in)
    elif noise_mode == 'pepper':
        img_in = add_pepper_noise(img_in)

    # setting
    max_iter = 1000
    lamb_da = 0.01
    
    # work
    t_start = time.time()
    TVL1_result = TVL1(img_in, k_in, max_iter, lamb_da)
    t_end = time.time()
    
    # print duration
    duration = t_end - t_start
    print(f'    -> processing time = {duration:.2f} sec')

    # store image
    # imageio.imwrite(f'../MyDeblur_result/result_{TEST_PAT_SIZE}/TVL1_m_iter1000_lam0.010_{noise_mode}.png' , TVL1_result)
    imageio.imwrite(f'../result/exploration/{noise_mode}_noise/TVL1_m_iter1000_lam0.010_{noise_mode}.png' , TVL1_result)
    
    
########################################################################
def test_TVL2():    
    print ("//--------------------------------------------------------")
    print (f"start TVL2, TEST_PAT_SIZE: {TEST_PAT_SIZE}\n")
    
    # I/O
    img_in = np.asarray(Image.open(f'../data/blurred_image_{TEST_PAT_SIZE}/curiosity_medium.png'))
    k_in = np.asarray(Image.open('../data/kernel/kernel_medium.png'))  
    if noise_mode == 'gaussian':
        img_in = add_gaussian_noise(img_in)
    elif noise_mode == 'pepper':
        img_in = add_pepper_noise(img_in)

    # setting
    max_iter = 1000
    lamb_da = 0.01
    to_linear = True

    # work
    t_start = time.time()
    TVL2_result = TVL2(img_in, k_in, max_iter, lamb_da, to_linear, gamma)
    t_end = time.time()
    
    # print duration
    duration = t_end - t_start
    print(f'    -> processing time = {duration:.2f} sec')

    # store image
    # imageio.imwrite(f'../MyDeblur_result/result_{TEST_PAT_SIZE}/TVL2_m_iter1000_lam0.010_{noise_mode}.png' , TVL2_result)
    imageio.imwrite(f'../result/exploration/{noise_mode}_noise/TVL2_m_iter1000_lam0.010_{noise_mode}.png' , TVL2_result)

########################################################################
def test_TVpoisson():    
    print ("//--------------------------------------------------------")
    print (f"start TVpoisson, TEST_PAT_SIZE: {TEST_PAT_SIZE}\n")
    
    # I/O
    img_in = np.asarray(Image.open(f'../data/blurred_image_{TEST_PAT_SIZE}/curiosity_medium.png'))
    k_in = np.asarray(Image.open('../data/kernel/kernel_medium.png'))  
    if noise_mode == 'gaussian':
        img_in = add_gaussian_noise(img_in)
    elif noise_mode == 'pepper':
        img_in = add_pepper_noise(img_in)

    # setting
    max_iter = 1000
    lamb_da = 0.01
    to_linear = False

    # work
    t_start = time.time()
    TVpoisson_result = TVpoisson(img_in, k_in, max_iter, lamb_da, to_linear, gamma)
    t_end = time.time()

    # store image
    # imageio.imwrite(f'../MyDeblur_result/result_{TEST_PAT_SIZE}/TVpoisson_m_iter1000_lam0.010_{noise_mode}.png' , TVpoisson_result)
    imageio.imwrite(f'../result/exploration/{noise_mode}_noise/TVpoisson_m_iter1000_lam0.010_{noise_mode}.png' , TVpoisson_result)
    
    # print duration
    duration = t_end - t_start
    print(f'    -> processing time = {duration:.2f} sec')

if __name__ == '__main__':

    ## (1) Wiener part
    
    test_Wiener_deconv()
    
    
    ## (2) RL part
    
    test_RL()
    
    
    ## (3) BRL part
    
    test_BRL()

    ## (4) Total variation part
    
    test_TVL1() # already done function for ref !
    test_TVL2()
    test_TVpoisson()
    