
from __future__ import absolute_import, division, print_function
import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import cv2
import torch
from torchvision import transforms, datasets
import numpy as np
import networks
from layers import disp_to_depth
from imageio import mimread,imsave
import imageio
# from utils import download_model_if_doesnt_exist
import pdb
from evaluate_depth import STEREO_SCALE_FACTOR


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')
    parser.add_argument('--video_path', type=str,
                        help='path to a test image or folder of images',
                        default='assets_mp4')
    
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',
                        )
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="mp4")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument("--pred_metric_depth",
                        help='if set, predicts metric depth instead of disparity.',
                        action='store_true')

    return parser.parse_args()

def test_video(args):
    
    """Function to predict for a single image or folder of images
    """
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.pred_metric_depth and "stereo" not in args.model_name:
        print("Warning: The --pred_metric_depth flag only makes sense for stereo-trained dataset "
              "models. For mono-trained models, output depths will not in metric space.")
    # download_model_if_doesnt_exist(args.model_name)
    # model_path = os.path.join("models", args.model_name)
    model_path = args.model_name
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(50, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)
        
    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()
    # FINDING INPUT IMAGES
    if os.path.isfile(args.video_path):
        # Only testing on a single image
        paths = [args.image_path]
        output_directory = os.path.dirname(args.video_path)
    elif os.path.isdir(args.video_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(args.video_path, '*.{}'.format(args.ext)))
        output_directory = args.video_path
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))
    
    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, video_path in enumerate(paths):

            if 'disp' in video_path:
                # don't try to predict disparity for a disparity image!
                continue

            # Load video and preprocess
            video=cv2.VideoCapture(video_path)
            fps=video.get(cv2.CAP_PROP_FPS)
            size = (feed_width, feed_height)
            if video.isOpened():
                rval,frame=video.read()
            else:
                rval=False
            count=0
            frames = []
            while rval:
                count+=1
                print(count,frame.shape)
                origin_image = pil.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

                original_width, original_height = origin_image.size
                input_image = origin_image.resize((feed_width, feed_height), pil.LANCZOS)
                input_image = transforms.ToTensor()(input_image).unsqueeze(0)

                # PREDICTION
                input_image = input_image.to(device)
                features = encoder(input_image)
                outputs = depth_decoder(features)

                disp = outputs[("disp", 0)]
                if args.pred_metric_depth:
                    scaled_disp, disp = disp_to_depth(disp, 0.1, 100)
                disp_resized = torch.nn.functional.interpolate(
                    disp, (original_height, original_width), mode="bilinear", align_corners=False)
                # Saving colormapped depth image
                disp_resized_np = disp_resized.squeeze().cpu().numpy()
                vmax = np.percentile(disp_resized_np, 95)
                normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
                mapper = cm.ScalarMappable(norm=normalizer, cmap='rainbow')
                colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
                
                im = pil.fromarray(colormapped_im)
                frames.append(im)
                rval,frame=video.read()
            result = np.stack(frames) 
            imageio.mimsave(video_path[:-4]+'_disp.mp4', result, fps=fps)

           
    print('--> Done!')

if __name__ == '__main__':
    args = parse_args()
    test_video(args)
