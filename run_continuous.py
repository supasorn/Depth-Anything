import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm
import time
from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler, FileSystemEventHandler

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='/data/supasorn/img3dviewer/images')
parser.add_argument('--outdir', type=str, default='./vis_depth')
parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])

args = parser.parse_args()

def process_files(event):
    
    pipe_path = "/tmp/my_pipe"
    if not os.path.exists(pipe_path):
        os.mkfifo(pipe_path)

    os.makedirs(os.path.join(args.path, "depth"), exist_ok=True)

    print("process_files")
    while True:
        with open(pipe_path, "r") as pipe:
            message = pipe.readline()[:-1]  # Read a line and remove the newline character
            print("Received:", message)
            for filename in os.listdir(args.path):
                if not filename.endswith('.jpg') and not filename.endswith('.png'):
                    continue
                if os.path.exists(os.path.join(args.path, "depth", filename[:-4] + '.png')):
                    continue

                filename = os.path.join(args.path, filename)
                print("reading", filename)
                try:
                    raw_image = cv2.imread(filename)
                    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
                except:
                    continue
                
                h, w = image.shape[:2]
                
                image = transform({'image': image})['image']
                image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    depth = depth_anything(image)
                
                depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
                depth_encoded = (depth - depth.min()) / (depth.max() - depth.min()) * (2**24 - 1)
                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

                print(depth.shape, depth_encoded.shape)
                
                depth = depth.cpu().numpy().astype(np.uint8) 
                depth_encoded = depth_encoded.cpu().numpy()

                nh = 800
                nw = int(nh * depth_encoded.shape[1] / depth_encoded.shape[0])
                # Resize using cv2 (OpenCV)
                resized_depth_encoded = cv2.resize(depth_encoded, (nw, nh), interpolation=cv2.INTER_AREA)
                print(resized_depth_encoded.shape)

                def encodeValueToRGB(depth_encoded):
                    r = (depth_encoded / 256 / 256) % 256
                    g = (depth_encoded / 256) % 256
                    b = depth_encoded % 256
                    return np.stack([b, g, r], axis=-1).astype(np.uint8)

                depth_encoded = encodeValueToRGB(depth_encoded)
                resized_depth_encoded = encodeValueToRGB(resized_depth_encoded)

                depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
                
                filename = os.path.basename(filename)
                
                # cv2.imwrite(os.path.join(args.outdir, filename[:filename.rfind('.')] + '_depth.png'), depth)
                # cv2.imwrite(os.path.join(args.outdir, filename[:filename.rfind('.')] + '_depth_encoded.png'), depth_encoded)
                cv2.imwrite(os.path.join(args.path, "depth", filename[:-4] + '.png'), resized_depth_encoded)
                



if __name__ == '__main__':
    margin_width = 50
    caption_height = 60
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(args.encoder)).to(DEVICE).eval()
    
    total_params = sum(param.numel() for param in depth_anything.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))
    
    os.makedirs('depth', exist_ok=True)
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])


    event_handler = FileSystemEventHandler()
    event_handler.on_created = process_files
    event_handler.on_modified = process_files

    process_files(None)

    
