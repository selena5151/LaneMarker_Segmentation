import numpy as np
import cv2
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian, create_pairwise_bilateral

def apply_crf(image, probabilities, num_iterations=1):
    """
    Using CRF to Refine Semantic Segmentation mask
    param probabilities: The probability map with the shape (num_classes, H, W).
    param num_iterations: The number of iterations for CRF.
    return: The refined mask, with the shape (H, W).
    """
    num_classes, h, w = probabilities.shape
    d = dcrf.DenseCRF2D(w, h, num_classes)
    
    unary = unary_from_softmax(probabilities)
    d.setUnaryEnergy(unary)

    gaussian_pairwise_energy = create_pairwise_gaussian(sdims=(3, 3), shape=(h, w))
    d.addPairwiseEnergy(gaussian_pairwise_energy, compat=3)

    bilateral_pairwise_energy = create_pairwise_bilateral(sdims=(25, 25), schan=(13, 13, 13), img=image, chdim=2)
    d.addPairwiseEnergy(bilateral_pairwise_energy, compat=10)
    
    Q = d.inference(num_iterations)
    
    refined_segmentation = np.argmax(Q, axis=0).reshape((h, w))
    
    return refined_segmentation