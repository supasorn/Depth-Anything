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
parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])

args = parser.parse_args()

def process_files(event):
    
    pipe_path = "/tmp/my_pipe"
    if not os.path.exists(pipe_path):
        os.mkfifo(pipe_path)

    os.makedirs(os.path.join(args.path, "depth"), exist_ok=True)

    while True:
        with open(pipe_path, "r") as pipe:
            message = pipe.readline()[:-1]  # Read a line and remove the newline character
            print("Received:", message)
            for filename in os.listdir(args.path):
                kind = ""
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    kind = "image"
                if filename.endswith('.mp4'):
                    kind = "video"

                if kind == "":
                    continue
                if kind == "image" and os.path.exists(os.path.join(args.path, "depth", filename[:-4] + '.png')):
                    continue

                filename = os.path.join(args.path, filename)
                print("reading", filename)

                
                if kind == "image":
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

                    # nh = 800
                    # nw = int(nh * depth_encoded.shape[1] / depth_encoded.shape[0])
                    # Resize using cv2 (OpenCV)
                    # resized_depth_encoded = cv2.resize(depth_encoded, (nw, nh), interpolation=cv2.INTER_AREA)
                    # resized_depth_encoded = depth_encoded

                    # print(resized_depth_encoded.shape)

                    def encodeValueToRGB(depth_encoded):
                        r = (depth_encoded / 256 / 256) % 256
                        g = (depth_encoded / 256) % 256
                        b = depth_encoded % 256
                        return np.stack([b, g, r], axis=-1).astype(np.uint8)

                    depth_encoded = encodeValueToRGB(depth_encoded)
                    # resized_depth_encoded = encodeValueToRGB(resized_depth_encoded)

                    # depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
                    
                    filename = os.path.basename(filename)
                    
                    # cv2.imwrite(os.path.join(args.outdir, filename[:filename.rfind('.')] + '_depth.png'), depth)
                    # cv2.imwrite(os.path.join(args.outdir, filename[:filename.rfind('.')] + '_depth_encoded.png'), depth_encoded)
                    # cv2.imwrite(os.path.join(args.path, "depth", filename[:-4] + '.png'), resized_depth_encoded)
                    cv2.imwrite(os.path.join(args.path, "depth", filename[:-4] + '.png'), depth_encoded)
                elif kind == "video":
                    if "processed" in filename:
                        continue

                    raw_video = cv2.VideoCapture(filename)
                    frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
                        raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
                    output_width = frame_width * 2 

                    filename_base = os.path.basename(filename)
                    output_path = os.path.join(args.path, filename_base[:filename_base.rfind('.')])

                    if os.path.exists(output_path + "_processed.mp4") or os.path.exists(os.path.join(output_path, f"0000.npy")):
                        continue
                    os.makedirs(output_path, exist_ok=True)

                    count = 0
                    while raw_video.isOpened():
                      ret, raw_frame = raw_video.read()
                      print(count)
                      if not ret:
                        break

                      frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB) / 255.0

                      frame = transform({'image': frame})['image']
                      frame = torch.from_numpy(frame).unsqueeze(0).to(DEVICE)

                      with torch.no_grad():
                        depth = depth_anything(frame)

                      depth = depth[0]
                      depth = depth.cpu().numpy()
                      np.save(os.path.join(output_path, f"{count:04d}.npy"), depth)
                      cv2.imwrite(os.path.join(output_path, f"{count:04d}.png"), raw_frame)
                      count += 1

                    raw_video.release()

                    os.system("cd /data/supasorn/opticalflow_ce_standalone; ./process_video " + output_path)
                    os.system(f"ffmpeg -i {output_path}_processed.mp4 -i {filename} -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 {output_path}_processed_audio.mp4")
                    os.system(f"rm {output_path}_processed.mp4")
                    os.system(f"mv {output_path}_processed_audio.mp4 {output_path}_processed.mp4")
                    
                



if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(args.encoder)).to(DEVICE).eval()
    
    total_params = sum(param.numel() for param in depth_anything.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))
    
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
    process_files(None)

    
