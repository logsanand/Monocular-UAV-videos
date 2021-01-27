# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

import networks
from layers_3d import disp_to_depth
from utils import download_model_if_doesnt_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',
                        choices=[
                            "mono_test",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320"])
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')

    return parser.parse_args()


def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("models", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")
    ##pose_path = os.path.join(model_path, "pose.pth")
    ##pose_encoder_path=os.path.join(model_path,"pose_encoder.pth")
    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    #encoder1 = networks.ResnetEncoder_low(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)
    ##pose_encoder=networks.ResnetEncoder(18,False)
    ##loaded_dict_poseenc=torch.load(pose_encoder_path,map_location=device)
    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()
    ##pose_encoder.load_state_dict(loaded_dict_poseenc)
    ##pose_encoder.to(device)
    ##pose_encoder.eval()
    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder_3d(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    ##print("   Loading pretrained pose")
    ##pose = networks.PoseDecoder(
        ##num_ch_enc=pose_encoder.num_ch_enc, num_input_features=1,num_frames_to_predict_for=2)

    ##loaded_dict_pose = torch.load(pose_path, map_location=device)
    ##pose.load_state_dict(loaded_dict_pose)

    ##pose.to(device)
    ##pose.eval()
    # FINDING INPUT IMAGES
    if os.path.isfile(args.image_path):
        # Only testing on a single image
        paths = [args.image_path]
        output_directory = os.path.dirname(args.image_path)
    elif os.path.isdir(args.image_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
        output_directory = os.path.join(args.image_path,"disparity_image")
        if not os.path.exists(output_directory):
           os.mkdir(output_directory)
        
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image1 = input_image.resize((320, 176), pil.LANCZOS)
            input_image1 = transforms.ToTensor()(input_image1).unsqueeze(0)
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)
            #input_image = torch.unsqueeze(input_image,2)
            #input_image1 = torch.unsqueeze(input_image1,2)


            # PREDICTION
            input_image = input_image.to(device)
            features1 = encoder(input_image)
            features2 = encoder(input_image)
            feature_concat1=torch.cat((torch.unsqueeze(features1[0],2),torch.unsqueeze(features2[0],2)),2)
            feature_concat2=torch.cat((torch.unsqueeze(features1[1],2),torch.unsqueeze(features2[1],2)),2)
            feature_concat3=torch.cat((torch.unsqueeze(features1[2],2),torch.unsqueeze(features2[2],2)),2)
            feature_concat4=torch.cat((torch.unsqueeze(features1[3],2),torch.unsqueeze(features2[3],2)),2)
            feature_concat5=torch.cat((torch.unsqueeze(features1[4],2),torch.unsqueeze(features2[4],2)),2)
            #feature_concat1=(features1[0])
            #feature_concat1=torch.cat((features1[0],features2[0]),2)
            #feature_concat2=torch.cat((features1[1],features2[0]),1)
            #feature_concat3=torch.cat((features1[2],features2[1]),1)
            #feature_concat4=torch.cat((features1[3],features2[2]),1)
            #feature_concat5=torch.cat((features1[4],features2[3]),1)
            feature_concat=[]
            feature_concat.append(feature_concat1)
            feature_concat.append(feature_concat2)
            feature_concat.append(feature_concat3)
            feature_concat.append(feature_concat4)
            feature_concat.append(feature_concat5)
            outputs = depth_decoder(feature_concat)
            ##axis,trans=pose(features)
            ##print(axis)
            ##print(trans)
            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving numpy file
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
            name_dest_npy2 = os.path.join(output_directory, "{}_disp2.npy".format(output_name))
            scaled_disp, _ = disp_to_depth(disp, 0.1, 500)#0.1,100
            np.save(name_dest_npy, scaled_disp.cpu().numpy())
            np.save(name_dest_npy2,disp.cpu().numpy())
            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
            im.save(name_dest_im)

            print("   Processed {:d} of {:d} images - saved prediction to {}".format(
                idx + 1, len(paths), name_dest_im))

    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
