from __future__ import annotations

from bonelab.util.vtk_util import vtkImageData_to_numpy
from bonelab.util.vtk_util import numpy_to_vtkImageData
from glob import glob
import numpy as np
import os
import vtk
from blpytorchlightning.tasks.SegmentationTask import SegmentationTask
from monai.networks.nets.unet import UNet
from monai.networks.nets.attentionunet import AttentionUnet
from monai.networks.nets.unetr import UNETR
from monai.networks.nets.basic_unetplusplus import BasicUNetPlusPlus
from torch.nn import CrossEntropyLoss

import argparse
import torch
import yaml
import SimpleITK as sitk


TORCH_NNPACK_ENABLED=0

parser = argparse.ArgumentParser(description='retraining BML data')
parser.add_argument('-ld', '--log_dir', type=str, required=True,
                    help='path to the log directory')
parser.add_argument('-l', '--label', type=str, required=True,
                    help='label')
parser.add_argument('-v', '--versions', type=str, required=True,
                    help='versions as version1,version2,version3')
parser.add_argument('-i', '--image_path', type=str, required=True,
                    help='path to the input new image file (.nii format)')
parser.add_argument('-o', '--maskOUT_path', type=str, required=True,
                    help='path to the output directory')
args = parser.parse_args()

def create_unetplusplus_loss_function(loss_function):
    def unetplusplus_loss_function(y_hat_list, y):
        loss = 0
        for y_hat in y_hat_list:
            loss += loss_function(y_hat, y)
        return loss
    return unetplusplus_loss_function

def create_model(ref_hparams) -> torch.nn.Module():
    # create the model
    model_kwargs = {
        "spatial_dims": 3 if ref_hparams["is_3d"] else 2,
        "in_channels": ref_hparams["input_channels"],
        "out_channels": ref_hparams["output_channels"],
    }
    if ref_hparams["dropout"] < 0 or ref_hparams["dropout"] > 1:
        raise ValueError("dropout must be between 0 and 1")
    if ref_hparams.get("model_architecture") == "unet" or ref_hparams.get("model_architecture") is None:
        if len(ref_hparams["model_channels"]) < 2:
            raise ValueError("model channels must be sequence of integers of at least length 2")
        model_kwargs["channels"] = ref_hparams["model_channels"]
        model_kwargs["strides"] = [1 for _ in range(len(ref_hparams["model_channels"]) - 1)]
        model_kwargs["dropout"] = ref_hparams["dropout"]
        model = UNet(**model_kwargs)
    elif ref_hparams.get("model_architecture") == "attention-unet":
        if len(ref_hparams["model_channels"]) < 2:
            raise ValueError("model channels must be sequence of integers of at least length 2")
        model_kwargs["channels"] = ref_hparams["model_channels"]
        model_kwargs["strides"] = [1 for _ in range(len(ref_hparams["model_channels"]) - 1)]
        model_kwargs["dropout"] = ref_hparams["dropout"]
        model = AttentionUnet(**model_kwargs)
    elif ref_hparams.get("model_architecture") == "unet-r":
        if ref_hparams["image_size"] is None:
            raise ValueError("if model architecture set to `unet-r`, you must specify image size")
        if ref_hparams["is_3d"] and len(ref_hparams["image_size"]) != 3:
            raise ValueError("if 3D, image_size must be integer or length-3 sequence of integers")
        if not ref_hparams["is_3d"] and len(ref_hparams["image_size"]) != 2:
            raise ValueError("if not 3D, image_size must be integer or length-2 sequence of integers")
        model_kwargs["img_size"] = ref_hparams["image_size"]
        model_kwargs["dropout_rate"] = ref_hparams["dropout"]
        model_kwargs["feature_size"] = ref_hparams["unet_r_feature_size"]
        model_kwargs["hidden_size"] = ref_hparams["unet_r_hidden_size"]
        model_kwargs["mlp_dim"] = ref_hparams["unet_r_mlp_dim"]
        model_kwargs["num_heads"] = ref_hparams["unet_r_num_heads"]
        model = UNETR(**model_kwargs)
    elif ref_hparams.get("model_architecture") == "unet++":
        if len(ref_hparams["model_channels"]) != 6:
            raise ValueError("if model architecture set to `unet++`, model channels must be length-6 sequence of "
                             "integers")
        model_kwargs["features"] = ref_hparams["model_channels"]
        model_kwargs["dropout"] = ref_hparams["dropout"]
        model = BasicUNetPlusPlus(**model_kwargs)
    else:
        raise ValueError(f"model architecture must be `unet`, `attention-unet`, `unet++`, or `unet-r`, "
                         f"given {ref_hparams['model_architecture']}")

    model.float()

    return model
def get_task(log_dir, label, version):

    with open(os.path.join(log_dir, label, version, "hparams.yaml")) as f:
        ref_hparams = yaml.safe_load(f)

    model = create_model(ref_hparams)

    loss_function = CrossEntropyLoss()
    if ref_hparams.get("model_architecture") == "unet++":
        loss_function = create_unetplusplus_loss_function(loss_function)

    checkpoint_path = glob(
        os.path.join(
            log_dir,
            label,
            version,
            "checkpoints",
            "*.ckpt"
        )
    )[0]
    print(f"Loading model and task from: {checkpoint_path}")

    # create the task
    task = SegmentationTask(
        model=model, loss_function=loss_function,
        learning_rate=ref_hparams["learning_rate"]
    )

    task = task.load_from_checkpoint(
        checkpoint_path,
        model=model, loss_function=loss_function,
        learning_rate=ref_hparams["learning_rate"]
    )

    return task


def crop_first_last(mask):
    # Get the size of the image
    mask_arr = sitk.GetArrayFromImage(mask)
    # Set all labels on the top 2 and bottom 2 slices to zero
    mask_arr[:2, :, :] = 0
    mask_arr[-2:, :, :] = 0
    cropped = sitk.GetImageFromArray(mask_arr)
    cropped.CopyInformation(mask)
    return cropped

def get_bone(mask):
    relabel = sitk.ChangeLabel(mask, changeMap={2:1})
    bone_mask = sitk.ConnectedComponent(relabel)

    bone_mask = sitk.BinaryMorphologicalOpening(bone_mask,[2,2,2])
    bone_mask = sitk.BinaryMorphologicalClosing(bone_mask,[2,2,2])
    bone_mask = sitk.RelabelComponent(bone_mask, minimumObjectSize=150000)
    bone_mask = sitk.ChangeLabel(bone_mask, changeMap={2:1,3:1,4:1,5:1,6:1,7:1,8:1,9:1})
    return bone_mask

def postpro(original_mask_path):
    #Read in mask from model
    mask = sitk.ReadImage(original_mask_path, sitk.sitkInt32)
    mask = crop_first_last(mask)

    #Get array from bone mask
    bone_mask = get_bone(mask)
    bone_mask2 = sitk.BinaryDilate(bone_mask,[2,2,2])
    bone_mask2 = sitk.BinaryMorphologicalClosing(bone_mask2, [4,4,4])
    bone_mask_arr = sitk.GetArrayFromImage(bone_mask2)

    #Get BML mask
    bml_mask = sitk.BinaryThreshold(mask,lowerThreshold=2)
    bml_mask = sitk.BinaryMorphologicalClosing(bml_mask, [3,3,3])
    bml_mask = sitk.ChangeLabel(bml_mask,changeMap={1:2})

    #Make sure all BML labels are within bone
    bml_mask = sitk.GetArrayFromImage(bml_mask)
    bml_mask = bone_mask_arr + bml_mask
    bml_mask = sitk.GetImageFromArray(bml_mask)
    bml_mask = sitk.ChangeLabel(bml_mask, changeMap={1:0,2:0,3:2})

    #remove small components
    bml_mask = sitk.ConnectedComponent(bml_mask)
    bml_mask = sitk.RelabelComponent(bml_mask, minimumObjectSize=500)
    bml_mask = sitk.ChangeLabel(bml_mask, changeMap={2:1,3:1,4:1,5:1,6:1,7:1,8:1})
    bml_mask = sitk.ChangeLabel(bml_mask, changeMap={1:2})

    return bml_mask, bone_mask


def calculate_DSC(true_mask,pred_mask):
    # Calculate Dice score
    intersection = np.sum(true_mask * pred_mask)
    dice_score = (2. * intersection) / (np.sum(true_mask) + np.sum(pred_mask))
    return dice_score

def calculate_precision(y_true, y_pred):
    # y_true and y_pred should be binary masks (0 or 1)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    return precision


def main():

    # Read the NIfTI image using VTK
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(args.image_path)
    reader.Update()
    image = vtkImageData_to_numpy(reader.GetOutput())
    image = image.astype(np.float32)
    image = np.flip(np.flip(image, axis=1), axis=2)

    # Normalize image intensities
    mean = np.mean(image)
    std = np.std(image)
    image_arr = (image - mean) / std

    # Scale intensities to the range of -1 to 1
    image_min = np.min(image_arr)
    image_max = np.max(image_arr)
    image_arr = (image_arr - image_min) / (image_max - image_min) * 2 - 1


    # parameters for loading the model and task
    log_dir = args.log_dir
    label = args.label      #"unet"
    versions = args.versions.split(',')

    # Initialize an empty list to store model outputs
    all_outputs = []

    # Load the trained model and task for each version
    for version in versions:
        current_task = get_task(log_dir, label, version)
        image_tensor = torch.tensor(image_arr).unsqueeze(0).unsqueeze(1).to(current_task.device)

        # Run inference on the entire volume
        current_task.model.eval()
        with torch.no_grad():
            output = current_task.model(image_tensor)

        # Append the model output to the list
        all_outputs.append(output)

    # average the model outputs
    average_output = torch.mean(torch.stack(all_outputs), dim=0).cpu().numpy()

    # convert the averaged output to a mask
    average_mask = average_output.argmax(axis=1).squeeze()

    # Flip the mask
    average_mask = np.flip(np.flip(average_mask, axis=1), axis=2)

    # Create a new VTK NIfTI writer
    writer = vtk.vtkNIFTIImageWriter()
    writer.SetFileName(args.maskOUT_path)
    writer.SetInputData(numpy_to_vtkImageData(average_mask))
    # Write the NIfTI file
    writer.Write()

    #POST PROCESSING
    original_mask_path = args.maskOUT_path
    bml_mask, bone_mask = postpro(original_mask_path)
    bml_mask = sitk.BinaryMorphologicalClosing(bml_mask)
    print()
    print("Results for file:",args.maskOUT_path)
    original_image = sitk.ReadImage(args.image_path)
    image_spacing = original_image.GetSpacing()
    bml_mask.SetSpacing(image_spacing)
    sitk.WriteImage(bml_mask, args.maskOUT_path[:-8] + 'BML.nii')


if __name__ == "__main__":
    main()