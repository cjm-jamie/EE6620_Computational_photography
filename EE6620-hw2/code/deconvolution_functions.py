''' Functions in deblur flow '''

import numpy as np
import cv2 as cv
from scipy import ndimage, signal
from scipy.signal import convolve2d
np.set_printoptions(threshold=np.inf)
import sys
import imageio
DBL_MIN = sys.float_info.min

########################################################
def kernal_preprocess(img_in, k_in, to_linear, gamma, rolling_window = True):
    """ kernal_preprocess
            Args:
                img_in (uint8 ndarray, shape(ch, height, width)): Blurred image
                k_in (uint8 ndarray, shape(height, width)): Blur kernel
                to_linear (bool): transform the photo to the linear domian before conducting deblur flow,
                                  and turn back to nonlinear domian after finishing deblur flow
                gamma (float): gamma for transfering linear domain and nonlinear domain

            Returns:
                k_pad (uint8 ndarray, shape(height, width)): Blur kernel after preprocessing
                
            Todo:
                kernal preprocess for Wiener deconvolution
    """
    if to_linear:
        k_in = np.power(k_in, gamma)

    # # Convert to floating point and normalize the image and kernel
    # k_float = k_in.astype(np.float64) / np.sum(k_in)
    
    # Get the shape of the image
    img_height, img_width = img_in.shape[1:]

    # Get the shape of the kernel
    k_height, k_width = k_in.shape

    # Pad the kernel with zeros to match the image resolution
    pad_height = img_height - k_height
    pad_width = img_width - k_width
    # k_pad = np.pad(k_float, ((0, pad_height), (0, pad_width)), mode='constant')
    k_pad = np.pad(k_in, ((0, pad_height), (0, pad_width)), mode='constant')

    print("Before rolling window!")

    if rolling_window:
        print("I'm doing rolling window!")
        # Roll the kernel for zero convention of the DFT
        k_pad = np.roll(k_pad, -(k_height // 2), axis=0)
        k_pad = np.roll(k_pad, -(k_width // 2), axis=1)

    return k_pad

########################################################
def deconv_Wiener(img_in, k_in, SNR_F, to_linear, gamma):
    """ Wiener deconvolution
            Args:
                img_in (uint8 ndarray, shape(ch, height, width)): Blurred image
                k_in (uint8 ndarray, shape(height, width)): Padded blur kernel
                SNR_F (float): Wiener deconvolution parameter
                to_linear (bool): transform the photo to the linear domian before conducting deblur flow,
                                  and turn back to nonlinear domian after finishing deblur flow
                gamma (float): gamma for transfering linear domain and nonlinear domain

            Returns:
                Wiener_result (uint8 ndarray, shape(ch, height, width)): Wiener-deconv image
                
            Todo:
                Wiener deconvolution
    """
    
    # Convert to floating point and normalize the image
    k_in = k_in.astype(np.float64) / np.sum(k_in)
    img_in = img_in.astype(np.float64) / 255.0
    
    if to_linear:
        for ch in range(img_in.shape[0]):
            img_in[ch] = np.power(img_in[ch], gamma)

    # Perform Fourier Transform on the kernel
    K_fft = np.fft.rfft2(k_in)
 
    # Initialize Wiener_result array to store the results
    Wiener_result = np.zeros_like(img_in, dtype=np.float64)

    # Iterate over each channel
    for ch in range(img_in.shape[0]):
        # Perform Fourier Transform on the image
        B_fft = np.fft.rfft2(img_in[ch])

        # Wiener deconvolution formula
        Wiener_filter = np.conj(K_fft) / (np.abs(K_fft) ** 2 + (1/SNR_F))
        Wiener_result_fft = B_fft * Wiener_filter

        # Store the result in the Wiener_result array
        Wiener_result[ch] = np.fft.irfft2(Wiener_result_fft)

    if to_linear:
        Wiener_result = np.power(Wiener_result, 1/gamma)   

    # Convert back to uint8 and return the result
    Wiener_result = np.round(np.clip(Wiener_result * 255, 0, 255)).astype(np.uint8)


    return Wiener_result


########################################################
def deconv_RL(img_in, k_in, max_iter, to_linear, gamma, boarder_mode='symmetric'):
    """ RL deconvolution
            Args:
                img_in (uint8 ndarray, shape(ch, height, width)): Blurred image
                k_in (uint8 ndarray, shape(height, width)): blur kernel
                max_iter (int): total iteration count
                to_linear (bool): transform the photo to the linear domian before conducting deblur flow,
                                  and turn back to nonlinear domian after finishing deblur flow
                gamma (float): gamma for transfering linear domain and nonlinear domain

            Returns:
                RL_result (uint8 ndarray, shape(ch, height, width)): RL-deblurred image
                
            Todo:
                RL deconvolution
    """

    # Convert to floating point and normalize the image and kernel
    img_float = img_in.astype(np.float64) / 255.0
    k_float = k_in.astype(np.float64) / np.sum(k_in)

    # convert to linear domain
    if to_linear:
        k_float = np.power(k_float, gamma)
        img_float = np.power(img_float, gamma)    

    # Initialize RL_result with the input image
    RL_result = np.zeros_like(img_in, dtype=np.float64)
    k_star = np.flip(k_float)

    # Iterate RL deconvolution
    for ch in range(img_in.shape[0]):
        B = img_float[ch,:,:]
        I_t = img_float[ch].copy()
        for _ in range(max_iter):
            # Convolve RL_result with the kernel
            if boarder_mode == 'zero':
                I_t_padded = np.pad(I_t, [(k_in.shape[0]//2), (k_in.shape[0]//2)], mode='constant', constant_values=0)
            else:
                I_t_padded = np.pad(I_t, [(k_in.shape[0]//2), (k_in.shape[0]//2)], mode=boarder_mode)
            RL_convolved = convolve2d(I_t_padded, k_float, mode='valid')
            
            # Update RL_result according to RL deconvolution formula
            scale_factor = B / (RL_convolved + DBL_MIN)
            if boarder_mode == 'zero':
                scale_factor_padded = np.pad(scale_factor, [(k_in.shape[0]//2), (k_in.shape[0]//2)], mode='constant', constant_values=0)
            else:
                scale_factor_padded = np.pad(scale_factor, [(k_in.shape[0]//2), (k_in.shape[0]//2)], mode=boarder_mode)
            updated_scale_factor = convolve2d(k_star, scale_factor_padded, mode='valid')
            
            # Update I^(t+1)
            I_t *= updated_scale_factor
            
        RL_result[ch] = I_t

    # convert back to nonlinear domain
    if to_linear:
        RL_result = np.power(RL_result, 1/gamma)               

    # Convert back to uint8
    RL_result = np.round(np.clip(RL_result * 255, 0, 255)).astype(np.uint8)

    return RL_result


########################################################
def RL_energy(img_in, k_in, I_in, boarder_mode='symmetric'):
    """ RL Energy
            Args:
                img_in (uint8 ndarray, shape(ch, height, width)): Blurred image
                k_in (uint8 ndarray, shape(height, width)): Blur kernel
                I_in (uint8 ndarray, shape(ch, height, width)): Deblurred image
                
            Returns:
                RL_energy (float): RL_energy
                
            Todo:
                Calculate RL energy
    """

    # Convert to floating point and normalize the image, kernel, and deblurred image
    img_float = img_in.astype(np.float32) / 255.0
    I_float = I_in.astype(np.float32) / 255.0
    k_float = k_in.astype(np.float32) / np.sum(k_in)
    
    energy = 0
    # Compute RL energy
    for ch in range(img_in.shape[0]):
        I_padded = np.pad(I_float[ch], [(k_in.shape[0]//2), (k_in.shape[0]//2)], mode=boarder_mode)
        convolved = convolve2d(I_padded, k_float, mode='valid')
        energy_ch = np.sum(convolved - img_float[ch] * np.log(convolved))
        energy += energy_ch

    return energy

########################################################

def gradient_BRL_energy(I_t, sigma_r, rk, boarder_mode='symmetric'):
    """ Compute gradient of BRL energy

    Args:
        I_t (float64 ndarray, shape(height, width)): Deblurred image at iteration t
        sigma_r (float): BRL parameter
        rk (int): BRL parameter
    Returns:
        gradient (float64 ndarray, shape(height, width)): Gradient of BRL energy
    
    Todo:
        get gradient_BRL for deconv_BRL()
    """

    # Compute spatial standard deviation sigma_s
    r_omega = rk // 2
    sigma_s = np.power(r_omega/3, 2)

    k, l = np.mgrid[ -r_omega :r_omega + 1, -r_omega : r_omega + 1]
    spatial_term = np.exp(-(k**2 + l**2) / (2 * sigma_s))

    padded_It = np.pad(I_t, [(r_omega), (r_omega)], mode=boarder_mode)

    # Compute gradient of BRL energy
    gradient = np.zeros_like(I_t)
    for y in range(I_t.shape[0]):
        for x in range(I_t.shape[1]):
            intensity_difference = (padded_It[y:y+(rk+1), x:x+(rk+1)] - I_t[y, x]) * -1
            range_term = np.exp(- (intensity_difference ** 2) / (2 * sigma_r))
            gradient[y, x] = 2 * np.sum (spatial_term * range_term * intensity_difference / sigma_r)

    return gradient

########################################################
def deconv_BRL(img_in, k_in, max_iter, lamb_da, sigma_r, rk, to_linear, gamma, boarder_mode='symmetric'):
    """ BRL deconvolution
            Args:
                img_in (uint8 ndarray, shape(ch, height, width)): Blurred image
                k_in (uint8 ndarray, shape(height, width)): Blur kernel
                max_iter (int): Total iteration count
                lamb_da (float): BRL parameter
                sigma_r (float): BRL parameter
                rk (int): BRL parameter
                to_linear (bool): transform the photo to the linear domian before conducting deblur flow,
                                  and turn back to nonlinear domian after finishing deblur flow
                gamma (float): gamma for transfering linear domain and nonlinear domain

            Returns:
                BRL_result (uint8 ndarray, shape(ch, height, width)): BRL-deblurred image
                
            Todo:
                BRL deconvolution
    """ 
      
    # Convert to floating point and normalize the image and kernel
    img_float = img_in.astype(np.float64) / 255.0
    k_float = k_in.astype(np.float64) / np.sum(k_in)

    # convert to linear domain
    if to_linear:
        k_float = np.power(k_float, gamma)
        img_float = np.power(img_float, gamma)     

    # Initialize BRL_result with the input image
    BRL_result = np.zeros_like(img_float, dtype=np.float64)
    k_star = np.flip(k_float)

    # Iterate BRL deconvolution
    for ch in range(img_float.shape[0]):
        B = img_float[ch,:,:]
        I_t = img_float[ch].copy()
        for _ in range(max_iter):
            # Convolve BRL_result with the kernel
            I_t_padded = np.pad(I_t, [(k_in.shape[0]//2), (k_in.shape[0]//2)], mode=boarder_mode)
            RL_convolved = convolve2d(I_t_padded, k_float, mode='valid')

            # Update BRL_result according to RL deconvolution formula
            scale_factor = B / (RL_convolved + DBL_MIN)
            scale_factor_padded = np.pad(scale_factor, [(k_in.shape[0]//2), (k_in.shape[0]//2)], mode=boarder_mode)
            updated_scale_factor = convolve2d(k_star, scale_factor_padded, mode='valid')
            
            # calculate energy gradient of deblurred image
            grad_BRL = gradient_BRL_energy(I_t, sigma_r, rk)
            BRL_coefficent = I_t / (1 + lamb_da * grad_BRL)

            # Update I^(t+1)
            I_t = BRL_coefficent * updated_scale_factor
            
        BRL_result[ch] = I_t

    if to_linear:
        BRL_result = np.power(BRL_result, 1/gamma)               

    # Convert back to uint8
    BRL_result = np.round(np.clip(BRL_result * 255, 0, 255)).astype(np.uint8)
   
    return BRL_result
    


########################################################
def BRL_EB(I_in, sigma_r, rk, boarder_mode='symmetric'):
    """ BRL Edge-preserving regularization term
            Args:
                I_in (uint8 ndarray, shape(ch, height, width)): Deblurred image
                sigma_r (float): BRL parameter
                rk (int): BRL parameter
                
            Returns:
                EB (float ndarray, shape(ch)): Edge-preserving regularization term
                
            Todo:
                Calculate BRL Edge-preserving regularization term
    """
    # Convert to floating point and normalize the deblurred image
    I_float = I_in.astype(np.float32) / 255.0

    # Compute spatial standard deviation sigma_s
    r_omega = rk // 2
    sigma_s = (r_omega / 3) ** 2

    # Initialize EB
    EB = np.zeros((I_float.shape[0], 1)).astype(np.float64)

    k, l = np.mgrid[ -r_omega :r_omega + 1, -r_omega : r_omega + 1]
    spatial_term = np.exp(-(k**2 + l**2) / (2 * sigma_s))

    # Iterate over each channel
    for ch in range(I_float.shape[0]):
        # Compute EB for the current channel
        EB_ch = 0.0
        padded_Iin = np.pad(I_float[ch,:,:], [(r_omega), (r_omega)], mode=boarder_mode)
        for y in range(I_float.shape[1]):
            for x in range(I_float.shape[2]):
                intensity_difference = (padded_Iin[y:y+(rk+1), x:x+(rk+1)] - I_float[ch, y, x]) * -1
                range_term = np.exp(- (intensity_difference ** 2) / (2 * sigma_r))
                EB_ch += np.sum(spatial_term * (1 - range_term))
        EB[ch] = EB_ch

    return EB

########################################################
def BRL_energy(img_in, k_in, I_in, lamb_da, sigma_r, rk, EB, boarder_mode='symmetric'):
    """ BRL Energy
            Args:
                img_in (uint8 ndarray, shape(ch, height, width)): Blurred image
                k_in (uint8 ndarray, shape(height, width)): Blur kernel
                I_in (uint8 ndarray, shape(ch, height, width)): Deblurred image
                lamb_da (float): BRL parameter
                sigma_r (float): BRL parameter
                rk (int): BRL parameter
                EB (float ndarray, shape(ch)): Edge-preserving regularization term

            Returns:
                BRL_energy (float): BRL_energy
                
            Todo:
                Calculate BRL energy
    """
    # Convert to floating point and normalize the image, kernel, and deblurred image
    img_float = img_in.astype(np.float32) / 255.0
    I_float = I_in.astype(np.float32) / 255.0
    k_float = k_in.astype(np.float32) / np.sum(k_in)
    
    energy = 0
    # Compute BRL energy
    for ch in range(img_in.shape[0]):
        I_padded = np.pad(I_float[ch], [(k_in.shape[0]//2), (k_in.shape[0]//2)], mode=boarder_mode)
        convolved = convolve2d(I_padded, k_float, mode='valid')
        energy_ch = np.sum(convolved - img_float[ch] * np.log(convolved))
        energy += (energy_ch + lamb_da * EB[ch])

    return float(energy)