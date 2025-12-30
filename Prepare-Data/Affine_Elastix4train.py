import SimpleITK as sitk
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.io as sio

def ncc(fix_im, mov_im):
    """
    Compute the Normalized Cross Correlation (NCC) between two images.
    
    Parameters:
    fix_im (numpy.ndarray): The fixed/reference image.
    mov_im (numpy.ndarray): The moving/target image.
    
    Returns:
    float: The NCC value between the two images.
    """
    fix_mean = np.mean(fix_im)
    mov_mean = np.mean(mov_im)
    

    fix_centered = fix_im - fix_mean
    mov_centered = mov_im - mov_mean
    
    numerator = np.sum(fix_centered * mov_centered)
    denominator = np.sqrt(np.sum(fix_centered ** 2)) * np.sqrt(np.sum(mov_centered ** 2))

    return numerator / denominator

    
def apply_tfmmap(moving_sitk, tfmmap):
    """
    Applies a given transform map to an image 
    """
    Elastix = sitk.TransformixImageFilter()
    Elastix.SetTransformParameterMap(tfmmap)
    Elastix.SetMovingImage(moving_sitk)
    Elastix.Execute()
    return Elastix.GetResultImage()


# How to use 

# Both images need to be sampled to the same spacing and size to avoid errors

# Fixed - SITK img of the reference image
# Moving - STIK img of the image to transform

# Moved,TFmap = multistep_registration(Fixed,Moving)


def multistep_registration(fixed, moving):
    """
    Perfroms rigid registration between fixed and moving. 
    Parameters
    ---------------
    r_type (str) : either affine or similarity (adds stretching)
    metric_type (str) : either ncc, mu, nmi, msd or multi (MI and Bending Energy)
    
    ---------------
    Returns the registered moving image and the ParamterMap.
    
    """
    
    #elastixImageFilter = sitk.SimpleElastix()
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixed)
    elastixImageFilter.SetMovingImage(moving)
    
    parameterMapVector = sitk.VectorOfParameterMap()
    parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
    elastixImageFilter.SetParameterMap(parameterMapVector)
    
    output = elastixImageFilter.Execute()
    transformParameterMap = elastixImageFilter.GetTransformParameterMap()[0]    


    return output, transformParameterMap


add_f = "/path/to/XMorpher-registration/Prepare-Data/4_resampled_128/fixed/"
add_m= "/path/to/XMorpher-registration/Prepare-Data/4_resampled_128/moving/"
lab_moving = "/path/to/XMorpher-registration/Prepare-Data/4_resampled_128/labelM/"
save_add = "/path/to/XMorpher-registration/data/train_unlabeled_unlabeled/"
save_add2 = "/path/to/XMorpher-registration/data/train_labeled_unlabeled/"

tranform_param_add = "/path/to/XMorpher-registration/Prepare-Data/"

i=0

ncc_metric =  []

for fix_im in sorted(os.listdir(add_f)):
    if fix_im!='.ipynb_checkpoints':
        new_add = add_f + fix_im
        fixed = sitk.ReadImage(new_add)
        fixed_array = sitk.GetArrayFromImage(fixed).transpose((2, 1, 0))


        for moving_im in sorted(os.listdir(add_m)):
            if moving_im!=fix_im and moving_im!='.ipynb_checkpoints':
                
                ad = add_m+moving_im
                moving = sitk.ReadImage(ad)
                moving_array = sitk.GetArrayFromImage(moving).transpose((2, 1, 0))

                l_m = lab_moving + moving_im 
                label = sitk.ReadImage(l_m)

                f = fix_im.split('.')[0]
                m = moving_im.split('.')[0]


                add_moved = save_add+f'{f}_{m}.mat'
                add_moved_nii = save_add+f'{f}_{m}.nii.gz'
                
               
                moved, tfmap =  multistep_registration(fixed, moving)
                moved_array = sitk.GetArrayFromImage(moved).transpose((2, 1, 0))


                movedlabel = apply_tfmmap(label,tfmap)
                #sitk.WriteImage(movedlabel, lab_m)
                
                movedlabel = sitk.GetArrayFromImage(movedlabel)
                movedlabeltr= movedlabel.round() #Rounds to the nearest integer
                movedlabeltr = np.clip(movedlabeltr, 0, 7)
                tr = torch.nn.Threshold(0.0, 0.0, inplace=False)
                movedlabeltr= tr(torch.Tensor(movedlabeltr)).numpy()
                movedlabeltrr = movedlabeltr.transpose((2, 1, 0))
                
                ncc_value0 = ncc(fixed_array, moving_array)
                ncc_value = ncc(fixed_array, moved_array)
                print(f"NCC between fixed and Moving (before affine): {ncc_value0:.4f}")
                print(f"NCC between fixed and moved (after affine): {ncc_value:.4f}")
                ncc_metric.append(ncc_value)
                
                # for train_unlabeled_unlabeled data pairs
                mat_dict = {
                'fix_img': fixed_array,
                'mov_img': moved_array}
                sio.savemat(add_moved, mat_dict)


                # # for train_labeled_unlabeled data pairs
                # mat_dict = {
                # 'fix_img': fixed_array,
                # 'mov_img': moved_array,
                # 'mov_lab': movedlabeltrr}
                # add_moved = save_add2+f'{f}_{m}.mat'


                sio.savemat(add_moved, mat_dict)
       
             

                i+=1
print(i)
print("The average NCC:", np.mean(ncc_metric))