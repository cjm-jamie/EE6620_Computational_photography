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

# research 2.2(b)
boarder_mode = 'symmetric'


########################################################################
def PSNR_UCHAR3(input_1, input_2, peak=255):
    [row,col,channel] = input_1.shape

    if input_1.shape != input_2.shape:
        print ("Warning!! Two image have different shape!!")
        print(input_1.shape)
        print(input_2.shape)
        return 0
    
    input_1 = input_1.astype('float')
    input_2 = input_2.astype('float')
    
    mse = ((input_1 - input_2)**2).sum() / float(row * col * channel)
    
    # print('mse: ', mse)
    if mse == 0.0:
        psnr = INF  # avoid divide zero case
    else:
        psnr = 10 * np.log10((255.0 ** 2)/mse)
    
    return psnr

########################################################################
def Evaluate_PSNR(psnr, duration, target_psnr=60.0):
    print(f'    -> processing time = {duration:.2f} sec, PSNR = {psnr} dB')
    
    if(psnr<target_psnr): 
        print('    -> status: \033[1;31;40m fail \033[0;0m ... QQ\n')
    else:
        print('    -> status: \033[1;32;40m pass \033[0;0m !!\n')   
    
    
    
########################################################################
def Evaluate_error(error, duration):
    print(f'    -> processing time = {duration:.2f} sec, error = {error:.4f} %')
    
    if(error>0.05): 
        print('    -> status: \033[1;31;40m fail \033[0;0m ... QQ\n')
    else:
        print('    -> status: \033[1;32;40m pass \033[0;0m !!\n') 
    
########################################################################
def read_image(file_path):
    img_in = np.asarray(Image.open(file_path))
    img_in = np.transpose(img_in, (2,0,1))

    return img_in

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
    img_golden = read_image(f'../golden/golden_{TEST_PAT_SIZE}/curiosity_medium/Wiener_m_SNRF150.0.png')
    
    # setting 
    SNR_F = 150.0
    to_linear = False
    
    # for research
    rolling_window = True

    # work
    t_start = time.time()
    k_in = kernal_preprocess(img_in, k_in, to_linear, gamma, rolling_window)
    Wiener_result = deconv_Wiener(img_in, k_in, SNR_F, to_linear, gamma)
    t_end = time.time()

    if rolling_window:
        # evaluate
        psnr = PSNR_UCHAR3(Wiener_result, img_golden)
        duration = t_end - t_start
        Evaluate_PSNR(psnr, duration)  

    # store image
    write_image(f'../result/result_{TEST_PAT_SIZE}/curiosity_medium/Wiener_m_SNRF{SNR_F}_rolling_window_{rolling_window}.png' , Wiener_result)




########################################################################
def test_RL_a():
    print ("//--------------------------------------------------------")
    print (f"start RL-(a) deconvolution, TEST_PAT_SIZE: {TEST_PAT_SIZE}\n")
    
    # I/O
    img_in = read_image(f'../data/blurred_image_{TEST_PAT_SIZE}/curiosity_small.png')
    k_in = np.asarray(Image.open('../data/kernel/kernel_small.png')) 
    img_golden = read_image(f'../golden/golden_{TEST_PAT_SIZE}/curiosity_small/RL_s_iter25.png')
    
    # setting 
    max_iter_RL = 25
    to_linear = True

    # work
    t_start = time.time()
    RL_result = deconv_RL(img_in, k_in, max_iter_RL, to_linear, gamma)
    t_end = time.time()

    # evaluate
    psnr = PSNR_UCHAR3(RL_result, img_golden)
    duration = t_end - t_start
    Evaluate_PSNR(psnr, duration)

    # store image
    write_image(f'../result/result_{TEST_PAT_SIZE}/curiosity_small/RL_s_iter{max_iter_RL}.png' , RL_result)

    
    
########################################################################
def test_RL_b():
    print ("//--------------------------------------------------------")
    print (f"start RL-(b) deconvolution, TEST_PAT_SIZE: {TEST_PAT_SIZE}\n")
    
    # I/O
    img_in = read_image(f'../data/blurred_image_{TEST_PAT_SIZE}/curiosity_medium.png')
    k_in = np.asarray(Image.open('../data/kernel/kernel_medium.png'))  
    img_golden = read_image(f'../golden/golden_{TEST_PAT_SIZE}/curiosity_medium/RL_m_iter50.png')
    

    # setting 
    max_iter_RL = 50
    to_linear = False

    
    # work   
    t_start = time.time()
    RL_result = deconv_RL(img_in, k_in, max_iter_RL, to_linear, gamma, boarder_mode)
    t_end = time.time()
    
    # evaluate
    psnr = PSNR_UCHAR3(RL_result, img_golden)
    duration = t_end - t_start
    Evaluate_PSNR(psnr, duration)

    # store image
    write_image(f'../result/result_{TEST_PAT_SIZE}/curiosity_medium/RL_m_iter{max_iter_RL}_{boarder_mode}.png' , RL_result)



    
    
########################################################################
def test_RL_energy():
    print ("//--------------------------------------------------------")
    print (f"start RL-energy function, TEST_PAT_SIZE: {TEST_PAT_SIZE}\n")
    
    # I/O
    blur_in = read_image(f'../data/blurred_image_{TEST_PAT_SIZE}/curiosity_small.png')
    k_in = np.asarray(Image.open('../data/kernel/kernel_small.png'))  
    img_in = read_image(f'../golden/golden_{TEST_PAT_SIZE}/curiosity_small/RL_s_iter25.png')
    
    energy_dict = np.load(f'../golden/energy_dict.npy',allow_pickle='TRUE').item()
    golden = energy_dict[f'{TEST_PAT_SIZE}_RL_a']
    
    # work
    t_start = time.time()
    energy = RL_energy(blur_in, k_in, img_in)
    t_end = time.time()

    # evaluate
    print(f'RL energy: {energy}, golden energy: {golden}')
    energy_dict_load[f'{TEST_PAT_SIZE}_RL_a'] = energy
    duration = t_end - t_start
    Evaluate_error( abs((energy-golden)/golden)*100, duration)

    
    
########################################################################   
def test_BRL_a():
    print ("//--------------------------------------------------------")
    print (f"start BRL-(a) deconvolution, TEST_PAT_SIZE: {TEST_PAT_SIZE}\n")
    
    # I/O
    img_in = read_image(f'../data/blurred_image_{TEST_PAT_SIZE}/curiosity_small.png')
    k_in = np.asarray(Image.open('../data/kernel/kernel_small.png'))
    img_golden = read_image(f'../golden/golden_{TEST_PAT_SIZE}/curiosity_small/BRL_s_iter20_rk6_si50.0_lam0.02.png')
    
    # setting
    max_iter_RL = 20
    rk = 6
    sigma_r = 50.0/255/255
    lamb_da = 0.02/255
    to_linear = False

    # work
    t_start = time.time()
    BRL_result = deconv_BRL(img_in, k_in, max_iter_RL, lamb_da, sigma_r, rk, to_linear, gamma)
    t_end = time.time()

    # evaluate
    psnr = PSNR_UCHAR3(BRL_result, img_golden)
    duration = t_end - t_start
    Evaluate_PSNR(psnr, duration, target_psnr=55.0)

    # store image
    sigma_r = sigma_r * 255 * 255
    lamb_da = lamb_da * 255
    write_image(f'../result/result_{TEST_PAT_SIZE}/curiosity_small/BRL_s_iter{max_iter_RL}_rk{rk}_si{sigma_r}_lam{lamb_da}.png' , BRL_result)



########################################################################   
def test_BRL_b():
    print ("//--------------------------------------------------------")
    print (f"start BRL-(b) deconvolution, TEST_PAT_SIZE: {TEST_PAT_SIZE}\n")
    
    # I/O
    img_in = read_image(f'../data/blurred_image_{TEST_PAT_SIZE}/curiosity_small.png')
    k_in = np.asarray(Image.open('../data/kernel/kernel_small.png'))  
    img_golden = read_image(f'../golden/golden_{TEST_PAT_SIZE}/curiosity_small/BRL_s_iter20_rk6_si50.0_lam0.05.png')
    
    # setting
    max_iter_RL = 20
    rk = 6
    sigma_r = 50.0/255/255
    lamb_da = 0.05/255
    to_linear = True

    # work
    t_start = time.time()
    BRL_result = deconv_BRL(img_in, k_in, max_iter_RL, lamb_da, sigma_r, rk, to_linear, gamma)
    t_end = time.time()
    
    # evaluate
    psnr = PSNR_UCHAR3(BRL_result, img_golden)
    duration = t_end - t_start
    Evaluate_PSNR(psnr, duration, target_psnr=55.0) 

    # store image
    sigma_r = sigma_r * 255 * 255
    lamb_da = lamb_da * 255
    write_image(f'../result/result_{TEST_PAT_SIZE}/curiosity_small/BRL_s_iter{max_iter_RL}_rk{rk}_si{sigma_r}_lam{lamb_da}.png' , BRL_result)




########################################################################   
def test_BRL_c():
    print ("//--------------------------------------------------------")
    print (f"start BRL-(c) deconvolution, TEST_PAT_SIZE: {TEST_PAT_SIZE}\n")
    
    # I/O
    img_in = read_image(f'../data/blurred_image_{TEST_PAT_SIZE}/curiosity_medium.png')
    k_in = np.asarray(Image.open('../data/kernel/kernel_medium.png'))  
    img_golden = read_image(f'../golden/golden_{TEST_PAT_SIZE}/curiosity_medium/BRL_m_iter50_rk12_si25.0_lam0.002.png')
    
    # setting
    max_iter_RL = 50
    rk = 12
    sigma_r = 25.0/255/255
    lamb_da = 0.002/255
    to_linear = False

    # work
    t_start = time.time()
    BRL_result = deconv_BRL(img_in, k_in, max_iter_RL, lamb_da, sigma_r, rk, to_linear, gamma)
    t_end = time.time()
    
    # evaluate
    psnr = PSNR_UCHAR3(BRL_result, img_golden)
    duration = t_end - t_start
    Evaluate_PSNR(psnr, duration, target_psnr=55.0)    

    # store image
    sigma_r = sigma_r * 255 * 255
    lamb_da = lamb_da * 255
    write_image(f'../result/result_{TEST_PAT_SIZE}/curiosity_medium/BRL_m_iter{max_iter_RL}_rk{rk}_si{sigma_r}_lam{lamb_da}.png' , BRL_result)
    
    
########################################################################   
def test_BRL_d():
    print ("//--------------------------------------------------------")
    print (f"start BRL-(d) deconvolution, TEST_PAT_SIZE: {TEST_PAT_SIZE}\n")
    
    # I/O
    img_in = read_image(f'../data/blurred_image_{TEST_PAT_SIZE}/curiosity_medium.png')
    k_in = np.asarray(Image.open('../data/kernel/kernel_medium.png'))  
    img_golden = read_image(f'../golden/golden_{TEST_PAT_SIZE}/curiosity_medium/BRL_m_iter50_rk12_si25.0_lam0.006.png')
    
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
    psnr = PSNR_UCHAR3(BRL_result, img_golden)
    
    # evaluate
    psnr = PSNR_UCHAR3(BRL_result, img_golden)
    duration = t_end - t_start
    Evaluate_PSNR(psnr, duration, target_psnr=55.0)

    # store image
    sigma_r = sigma_r * 255 * 255
    lamb_da = lamb_da * 255
    write_image(f'../result/result_{TEST_PAT_SIZE}/curiosity_medium/BRL_m_iter{max_iter_RL}_rk{rk}_si{sigma_r}_lam{lamb_da}.png' , BRL_result)
    
########################################################################
def test_BRL_energy():
    print ("//--------------------------------------------------------")
    print (f"start BRL-energy function, TEST_PAT_SIZE: {TEST_PAT_SIZE}\n")
    
    # I/O
    blur_in = read_image(f'../data/blurred_image_{TEST_PAT_SIZE}/curiosity_small.png')
    k_in = np.asarray(Image.open('../data/kernel/kernel_small.png'))  
    img_in = read_image(f'../golden/golden_{TEST_PAT_SIZE}/curiosity_small/BRL_s_iter20_rk6_si50.0_lam0.02.png')
    
    energy_dict = np.load('../golden/energy_dict.npy',allow_pickle='TRUE').item()
    golden = energy_dict[f'{TEST_PAT_SIZE}_BRL_a']
    

    # setting
    rk = 6
    sigma_r = 50.0/255/255
    lamb_da = 0.03/255

    # work
    t_start = time.time()
    energy_B = BRL_EB(img_in, sigma_r, rk) 
    energy = BRL_energy(blur_in, k_in, img_in, lamb_da, sigma_r, rk, energy_B)
    t_end = time.time()
    
    # evaluate
    print(f'BRL energy: {energy}, golden energy: {golden}')
    energy_dict_load[f'{TEST_PAT_SIZE}_BRL_a'] = energy
    duration = t_end - t_start
    Evaluate_error( abs((energy-golden)/golden)*100, duration)

        
    

########################################################################
def test_TVL1():    
    print ("//--------------------------------------------------------")
    print (f"start TVL1, TEST_PAT_SIZE: {TEST_PAT_SIZE}\n")
    
    # I/O
    img_in = np.asarray(Image.open(f'../data/blurred_image_{TEST_PAT_SIZE}/curiosity_medium.png'))
    k_in = np.asarray(Image.open('../data/kernel/kernel_medium.png'))  
    img_golden = np.asarray(Image.open(f'../golden/golden_{TEST_PAT_SIZE}/curiosity_medium/TVL1_m_iter1000_lam0.010.png'))
    
    # setting
    max_iter = 1000
    lamb_da = 0.01
    
    # work
    t_start = time.time()
    TVL1_result = TVL1(img_in, k_in, max_iter, lamb_da)
    t_end = time.time()
    
    
    # evaluate
    psnr = PSNR_UCHAR3(TVL1_result, img_golden)
    duration = t_end - t_start
    Evaluate_PSNR(psnr, duration)

    # store image
    imageio.imwrite(f'../result/result_{TEST_PAT_SIZE}/curiosity_medium/TVL1_m_iter1000_lam0.010.png' , TVL1_result)
    
    
########################################################################
def test_TVL2():    
    print ("//--------------------------------------------------------")
    print (f"start TVL2, TEST_PAT_SIZE: {TEST_PAT_SIZE}\n")
    
    # I/O
    img_in = np.asarray(Image.open(f'../data/blurred_image_{TEST_PAT_SIZE}/curiosity_medium.png'))
    k_in = np.asarray(Image.open('../data/kernel/kernel_medium.png'))  
    img_golden = np.asarray(Image.open(f'../golden/golden_{TEST_PAT_SIZE}/curiosity_medium/TVL2_m_iter1000_lam0.010.png'))
    
    # setting
    max_iter = 1000
    lamb_da = 0.01
    to_linear = False

    # work
    t_start = time.time()
    TVL2_result = TVL2(img_in, k_in, max_iter, lamb_da, to_linear, gamma)
    t_end = time.time()
    
    # evaluate
    psnr = PSNR_UCHAR3(TVL2_result, img_golden)
    duration = t_end - t_start
    Evaluate_PSNR(psnr, duration)

    # store image
    imageio.imwrite(f'../result/result_{TEST_PAT_SIZE}/curiosity_medium/TVL2_m_iter1000_lam0.010.png' , TVL2_result)
    

    
########################################################################
def test_TVpoisson():    
    print ("//--------------------------------------------------------")
    print (f"start TVpoisson, TEST_PAT_SIZE: {TEST_PAT_SIZE}\n")
    
    # I/O
    img_in = np.asarray(Image.open(f'../data/blurred_image_{TEST_PAT_SIZE}/curiosity_medium.png'))
    k_in = np.asarray(Image.open('../data/kernel/kernel_medium.png'))  
    img_golden = np.asarray(Image.open(f'../golden/golden_{TEST_PAT_SIZE}/curiosity_medium/TVpoisson_m_iter1000_lam0.010.png'))
    
    # setting
    max_iter = 1000
    lamb_da = 0.01
    to_linear = False

    # work
    t_start = time.time()
    TVpoisson_result = TVpoisson(img_in, k_in, max_iter, lamb_da, to_linear, gamma)
    t_end = time.time()

    # store image
    imageio.imwrite(f'../result/result_{TEST_PAT_SIZE}/curiosity_medium/TVpoisson_m_iter1000_lam0.010.png' , TVpoisson_result)
    
    # evaluate
    psnr = PSNR_UCHAR3(TVpoisson_result, img_golden)
    duration = t_end - t_start
    Evaluate_PSNR(psnr, duration)    
        
    
    
########################################################################
if __name__ == '__main__':

    ## (1) Wiener part
    
    test_Wiener_deconv()
    
    
    # (2) RL part
    
    test_RL_a()
    test_RL_b()
    test_RL_energy()
    
    
    # ## (3) BRL part
    
    test_BRL_a()
    test_BRL_b()
    test_BRL_c()
    test_BRL_d()
    test_BRL_energy()

    ## (4) Total variation part
    
    test_TVL1() # already done function for ref !
    test_TVL2()
    test_TVpoisson()
    
