import numpy as np
import cv2
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian, create_pairwise_bilateral

def apply_crf(image, probabilities, num_iterations=1):
    """
    使用 CRF 細化語意分割結果
    :param probabilities: 語意分割模型的概率圖，形狀為 (num_classes, H, W)
    :param num_iterations: CRF 的迭代次數
    :return: 細化後的分割結果，形狀為 (H, W)
    """
    # 創建 CRF 模型
    num_classes, h, w = probabilities.shape
    d = dcrf.DenseCRF2D(w, h, num_classes)
    
    # 將概率圖轉換為 CRF 的 unary energy
    unary = unary_from_softmax(probabilities)
    d.setUnaryEnergy(unary)
    
    # 創建 pairwise energy
    # 高斯核（低級特徵）
    gaussian_pairwise_energy = create_pairwise_gaussian(sdims=(3, 3), shape=(h, w))
    d.addPairwiseEnergy(gaussian_pairwise_energy, compat=3)
    
    # 雙邊核（高級特徵）
    bilateral_pairwise_energy = create_pairwise_bilateral(sdims=(25, 25), schan=(13, 13, 13), img=image, chdim=2)
    d.addPairwiseEnergy(bilateral_pairwise_energy, compat=10)
    
    # 執行推理
    Q = d.inference(num_iterations)
    
    # 重新格式化輸出
    refined_segmentation = np.argmax(Q, axis=0).reshape((h, w))
    
    return refined_segmentation