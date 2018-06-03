from skimage.measure import compare_ssim
import numpy as np

def get_ssim(original_img,changed_img):
    ssim = compare_ssim(np.array(original_img, dtype=np.float32),
                        np.array(changed_img, dtype=np.float32),
                        multichannel=True)
    return ssim