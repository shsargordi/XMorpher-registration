import SimpleITK as sitk
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import matplotlib.pyplot as plt
from torchmetrics.functional import dice
import scipy.io as sio
#from torchmetrics.functional import dice_score


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


#Affine registration of all images
add_fix = "/path/to/XMorpher-registration/Prepare-Data/4_resampled_128/fixed/"
add_moving = "/path/to/XMorpher-registration/Prepare-Data/4_resampled_128/moving/"

lab_fix = "/path/to/XMorpher-registration/Prepare-Data/4_resampled_128/labelF/"
lab_moving = "/path/to/XMorpher-registration/Prepare-Data/4_resampled_128/labelM/"

add_moved = "/path/to/XMorpher-registration/data/test/"


tranform_param_add = "/path/to/XMorpher-registration/Prepare-Data/"


i=0
dice_scores = []
total_dice =  []
str =  []
dice_s = 0
for fix_im in sorted(os.listdir(add_fix)):
    if fix_im!='.ipynb_checkpoints':
        new_add = add_fix + fix_im
        fixed = sitk.ReadImage(new_add)
        fixed_array = sitk.GetArrayFromImage(fixed).transpose((2, 1, 0))
        l = lab_fix + fix_im
        fixedlabel = sitk.ReadImage(l)
        fixedlabel_array = sitk.GetArrayFromImage(fixedlabel).transpose((2, 1, 0))
        
        for moving_im in sorted(os.listdir(add_moving)):
            if moving_im!=fix_im and moving_im!='.ipynb_checkpoints':
                ad = add_moving+moving_im
                l_m = lab_moving + moving_im 
                moving = sitk.ReadImage(ad)
                label = sitk.ReadImage(l_m)

                f = fix_im.split('.')[0]
                m = moving_im.split('.')[0]
                #moved_address = add_moved + f'{f}_{m}.nii.gz'
                #lab_m= moved_label_add + f'{f}_{m}.nii.gz'
                #str.append(f'{f}_{m}.nii.gz')
                
                moved_address = add_moved+f'{f}_{m}.mat'
                #lab_m= moved_label_add + f'{i}.nii.gz'

                moved, tfmap =  multistep_registration(fixed, moving)
                #sitk.WriteImage(moved, moved_address)
                moved_array = sitk.GetArrayFromImage(moved).transpose((2, 1, 0))

                movedlabel = apply_tfmmap(label,tfmap)
                #sitk.WriteImage(movedlabel, lab_m)
                
                movedlabel = sitk.GetArrayFromImage(movedlabel)
                movedlabeltr= movedlabel.round() #Rounds to the nearest integer
                movedlabeltr = np.clip(movedlabeltr, 0, 7)
                tr = torch.nn.Threshold(0.0, 0.0, inplace=False)
                movedlabeltr= tr(torch.Tensor(movedlabeltr)).numpy()
                movedlabeltrr = movedlabeltr.transpose((2, 1, 0))


                movela = sitk.GetArrayFromImage(label)
                movelatr= movela.round() #Rounds to the nearest integer
                movelatr = np.clip(movelatr, 0, 7)
                tr = torch.nn.Threshold(0.0, 0.0, inplace=False)
                movelatr= tr(torch.Tensor(movelatr)).numpy()
                #mmlabeltrrr = mmlabeltr.transpose((2, 1, 0))
                
                #label_image_save = sitk.GetImageFromArray(movedlabeltr)
                mat_dict = {
                'fix_img': fixed_array,
                'mov_img': moved_array,
                'fix_lab': fixedlabel_array,
                'mov_lab': movedlabeltrr}

                sio.savemat(moved_address, mat_dict)
                #sitk.WriteImage(label_image_save, lab_m)
                
                fixedlabel_np = sitk.GetArrayFromImage(fixedlabel)
                 
                molabeltr_tensor = torch.Tensor(movelatr).type(torch.int64)
                movedlabeltr_tensor = torch.Tensor(movedlabeltr).type(torch.int64)
                fixedlabel_tensor = torch.Tensor(fixedlabel_np).type(torch.int64)

                Dicescore0 = dice(preds=molabeltr_tensor, target=fixedlabel_tensor, ignore_index=0)
                Dicescore = dice(preds=movedlabeltr_tensor, target=fixedlabel_tensor, ignore_index=0)
                print('Dice score before affine', Dicescore0)
                print('Dice score after affine',Dicescore)
                
                dice_scores.append(Dicescore)
       
            
                print(str)
                print("Dice scores for each label:", dice_scores)
                print("Average Dice score:", np.mean(dice_scores))
                total_dice.append(np.mean(dice_scores))
                
                i+=1
print(i)