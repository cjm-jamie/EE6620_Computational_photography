import cv2
import numpy as np
import matplotlib.pyplot as plt

def fourier_transform_image(image_path):
    #讀取圖像
    img = cv2.imread(image_path,cv2.IMREAD_COLOR)
    
    #將圖像轉换為灰階
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.array(gray, dtype=np.float64) / 255

    #找到中間的fy=0行
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    #進行傅里葉轉換
    f = np.fft.fft(gray[crow, :])
    fshift = np.fft.fftshift(f)

    #取得頻譜
    spectrum_1d =np.log(np.abs(fshift))

    #計算對應的空間頻率
    freq_1d = np.fft.fftfreq(cols, d=1.0 /cols)
    freq_1d = np.fft.fftshift(freq_1d)
    freq_1d = freq_1d / cols * 2
    return spectrum_1d, freq_1d

if __name__ == '__main__' :
    image_paths = [
    './data/curiosity.png',
    './result/result_large/curiosity_medium/TVL1_m_iter1000_lam0.010.png',
    './result/result_large/curiosity_medium/TVL2_m_iter1000_lam0.010.png',
    './result/result_large/curiosity_medium/TVpoisson_m_iter1000_lam0.010.png',
    './result/result_large/curiosity_medium/BRL_m_iter50_rk12_si25.0_lam0.006.png',
    './result/result_large/curiosity_medium/RL_m_iter50.png',
    './result/result_large/curiosity_medium/Wiener_m_SNRF150.0.png',
    './data/blurred_image_large/curiosity_medium.png'
    ]

    # image_paths = [
    # './data/curiosity.png',
    # './result/exploration/gaussian_noise/TVL1_m_iter1000_lam0.010_gaussian.png',
    # './result/exploration/gaussian_noise/TVL2_m_iter1000_lam0.010_gaussian.png',
    # './result/exploration/gaussian_noise/TVpoisson_m_iter1000_lam0.010_gaussian.png',
    # './result/exploration/gaussian_noise/BRL_s_iter50_rk12_si25.0_lam0.006_gaussian.png',
    # './result/exploration/gaussian_noise/RL_s_iter50_gaussian.png',
    # './result/exploration/gaussian_noise/Wiener_m_SNRF150.0_rolling_window_gaussian.png',
    # './data/blurred_image_large/curiosity_medium.png'
    # ]

    fig, ax = plt.subplots(figsize=(10, 6))
    lines = []
    colors= plt.cm.rainbow(np.linspace(1,0, len(image_paths)))#生成彩虹色列表
    
    for i, image_path in enumerate(image_paths):
        spectrum,freqs = fourier_transform_image(image_path)
        line, = ax.plot(freqs, spectrum, color=colors[i], label=image_path.split('/')[-1])
        lines.append(line)
    
    ax.set_title('One-dimensional Spectrum (fy=0)')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Log Magnitude' )
    ax.set_xlim(0,1)#設置x軸範從到最大頻率
    ax.set_xticks([0,0.2,0.4,0.6, 0.8, 1.0])
    ax.set_xticklabels(['0', r'$\frac{1}{5}\pi$', r'$\frac{2}{5}\pi$', r'$\frac{3}{5}\pi$', r'$\frac{4}{5}\pi$', r'$\pi$'])
    ax.legend(lines, [line.get_label() for line in lines])
    plt.tight_layout()
    plt.savefig('./result/exploration/Experiment_2-2(c)_orginal.png', dpi=300, bbox_inches='tight')
    # plt.savefig('./result/exploration/Experiment_2-1.png', dpi=300, bbox_inches='tight')
    plt.show()
