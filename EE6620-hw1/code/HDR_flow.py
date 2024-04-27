''' HDR flow '''

from functools import partial

from HDR_functions import CameraResponseCalibration, WhiteBalance, \
                          GlobalTM, LocalTM, GaussianFilter, BilateralFilter, \
                          SaveImg, CompareHistograms


##### Test image: memorial #####
TestImage = 'my_image_delta'

print(f'---------- Test Image is {TestImage} ----------')
### Whole HDR flow ### 
print('Start to process HDR flow...')
# Camera response calibration
radiance = CameraResponseCalibration(f'../TestImage/{TestImage}', lambda_=50)
print('--Camera response calibration done')
# White balance
if TestImage == 'memorial':
    ktbw = (419, 443), (389, 401)
elif TestImage =='my_image_delta':
    ktbw = (2041, 2206), (2160, 2240)
elif TestImage =='my_image_room':
    ktbw = (3500, 3760), (1000, 1240)
radiance_wb = WhiteBalance(radiance, *ktbw)
print('--White balance done')
print('--Tone mapping')
# Global tone mapping
gtm_no_wb = GlobalTM(radiance, scale=1)  # without white balance
gtm = GlobalTM(radiance_wb, scale=1)     # with white balance
print('    Global tone mapping done')

# Local tone mapping with gaussian filter
ltm_filter = partial(GaussianFilter, N=15, sigma_s=100)
ltm_gaussian = LocalTM(radiance_wb, ltm_filter, scale=7)
print('    Local tone mapping with gaussian filter done')

# Local tone mapping with bilateral filter
ltm_filter = partial(BilateralFilter, N=15, sigma_s=100, sigma_r=0.8)
ltm_bilateral = LocalTM(radiance_wb, ltm_filter, scale=7)
print('    Local tone mapping with bilateral filter done')
print('Whole process done\n')

if TestImage == 'memorial':
    ### Save memorial result ###
    print('Mode: memorial')
    print('Saving results...')
    SaveImg(gtm_no_wb, f'../Result/{TestImage}_gtm_no_wb.png')
    SaveImg(gtm, f'../Result/{TestImage}_gtm.png')
    SaveImg(ltm_gaussian, f'../Result/{TestImage}_ltm_gau.png')
    SaveImg(ltm_bilateral, f'../Result/{TestImage}_ltm_bil.png')
    print('All results are saved\n')
else:
    ### Save my own result ###
    print(f'Mode: {TestImage}')
    print(f'Saving {TestImage} results...')
    SaveImg(gtm_no_wb, f'../MyHDR_result/{TestImage}/{TestImage}_gtm_no_wb.png')
    SaveImg(gtm, f'../MyHDR_result/{TestImage}/{TestImage}_gtm.png')
    SaveImg(ltm_gaussian, f'../MyHDR_result/{TestImage}/{TestImage}_ltm_gau.png')
    SaveImg(ltm_bilateral, f'../MyHDR_result/{TestImage}/{TestImage}_ltm_bil.png')
    print(f'All {TestImage} results are saved\n')
