import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from wimshow import imshow, waitKey, flipImages

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

import ptlflow
from ptlflow.utils import flow_utils
from ptlflow.utils.io_adapter import IOAdapter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str)
    parser.add_argument('--outdir', type=str, default='./vis_video_depth')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--process', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--optflow', action='store_true', help='run optical flow')
    
    args = parser.parse_args()

    if args.optflow:
        filename = os.path.basename(args.video_path)
        output_path = os.path.join(args.outdir, filename[:filename.rfind('.')] + '_video_depth')

        # model = ptlflow.get_model('rpknet', pretrained_ckpt='things')
        model = ptlflow.get_model('rapidflow', pretrained_ckpt='things')
        images = [
            # cv2.imread(f'{output_path}/0000.png'),
            # cv2.imread(f'{output_path}/0010.png')
            cv2.imread(f'images/car1.jpg'),
            cv2.imread(f'images/car2.jpg'),
        ]
        # A helper to manage inputs and outputs of the model
        io_adapter = IOAdapter(model, images[0].shape[:2])

        # inputs is a dict {'images': torch.Tensor}
        # The tensor is 5D with a shape BNCHW. In this case, it will have the shape:
        # (1, 2, 3, H, W)
        inputs = io_adapter.prepare_inputs(images)

        # Forward the inputs through the model
        predictions = model(inputs)

        # The output is a dict with possibly several keys,
        # but it should always store the optical flow prediction in a key called 'flows'.
        flows = predictions['flows']
        print(predictions.keys())

        # flows will be a 5D tensor BNCHW.
        # This example should print a shape (1, 1, 2, H, W).
        print(flows.shape)

        # Create an RGB representation of the flow to show it on the screen
        flow_rgb = flow_utils.flow_to_rgb(flows)
        # Make it a numpy array with HWC shape
        flow_rgb = flow_rgb[0, 0].permute(1, 2, 0)
        flow_rgb_npy = flow_rgb.detach().cpu().numpy()
        # OpenCV uses BGR format
        flow_bgr_npy = cv2.cvtColor(flow_rgb_npy, cv2.COLOR_RGB2BGR)

        # Convert flow tensor to numpy array
        flow_np = flows.detach().cpu().numpy()[0, 0]  # shape should be (2, H, W)
        W = flow_np.shape[2]
        H = flow_np.shape[1]
        xx, yy = np.meshgrid(np.arange(W), np.arange(H))

        map_x = (xx + flow_np[0]).astype(np.float32)
        map_y = (yy + flow_np[1]).astype(np.float32)

        warped_image = cv2.remap(images[1], map_x, map_y, interpolation=cv2.INTER_LINEAR)

        # Show on the screen
        imshow('image1', images[0], w=800)
        imshow('image2', images[1], w=800)
        imshow('warped', warped_image, w=800)
        flipImages('flip', warped_image, images[0], w=800)
        imshow('flow', flow_bgr_npy, w=800)
        waitKey()
        exit()

    if args.process:
        print('Processing depth')
        filename = os.path.basename(args.video_path)
        output_path = os.path.join(args.outdir, filename[:filename.rfind('.')] + '_video_depth')
        filenames = os.listdir(output_path)
        filenames = [os.path.join(output_path, filename) for filename in filenames if not filename.startswith('.')]
        filenames.sort()
        with open('depth.txt', 'w') as f:
            for k, fn in enumerate(filenames):
                if not fn.endswith('.npy'): continue
                print('Progress {:}/{:},'.format(k+1, len(filenames)), 'Processing', fn)
                depth = np.load(fn)
                f.write(f"{fn} {depth.min()} {depth.max()}\n")
                # print(depth.min(), depth.max())
                # depth = (depth - depth.min()) / (depth.max() - depth.min())
                # depth = (depth * 255).astype(np.uint8)
                # cv2.imwrite(fn.replace('.npy', '_d.png'), depth * 255)
                # imshow(depth, 'depth')
                # waitKey()
        with open('depth.txt', 'r') as f:
            lines = f.read().splitlines()
        
        mmin = 1e9
        mmax = -1e9

        for line in lines:
            fn, min_depth, max_depth = line.split()
            print('Processing', fn, min_depth, max_depth)
            mmin = min(float(min_depth), mmin)
            mmax = max(float(max_depth), mmax)

        for k, fn in enumerate(filenames):
            if not fn.endswith('.npy'): continue
            print('Progress {:}/{:},'.format(k+1, len(filenames)), 'Processing', fn)
            depth = np.load(fn)
            depth = (depth - mmin) / (mmax - mmin)
            depth = (depth * 255).astype(np.uint8)
            # dilate the depth image by kernel 5
            kernel = np.ones((9,9),np.uint8)
            depth = cv2.dilate(depth, kernel, iterations=1)

            depth = np.repeat(depth[:, :, None], 3, axis=-1)

            raw = cv2.imread(fn.replace('.npy', '.png'))
            raw = cv2.resize(raw, (depth.shape[1], depth.shape[0]))

            combined = cv2.hconcat([raw, depth])
            cv2.imwrite(fn.replace('.npy', '_c.png'), combined)
            # cv2.imwrite(fn.replace('.npy', '_d2.png'), depth)
        os.system(f"ffmpeg -y -i {output_path}/%04d_c.png -vcodec libx264 -crf 18 -pix_fmt yuv420p /data/supasorn/img3dviewer/videos/{filename}")
        exit()


    
    # margin_width = 50
    margin_width = 0

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

    if os.path.isfile(args.video_path):
        if args.video_path.endswith('txt'):
            with open(args.video_path, 'r') as f:
                lines = f.read().splitlines()
        else:
            filenames = [args.video_path]
    else:
        filenames = os.listdir(args.video_path)
        filenames = [os.path.join(args.video_path, filename) for filename in filenames if not filename.startswith('.')]
        filenames.sort()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    for k, filename in enumerate(filenames):
        print('Progress {:}/{:},'.format(k+1, len(filenames)), 'Processing', filename)
        
        raw_video = cv2.VideoCapture(filename)
        frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
        output_width = frame_width * 2 + margin_width
        
        filename = os.path.basename(filename)
        # output_path = os.path.join(args.outdir, filename[:filename.rfind('.')] + '_video_depth.mp4')
        # out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width, frame_height))
        # out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"avc1"), frame_rate, (output_width, frame_height))
        # create a temp folder for outputing png images
        output_path = os.path.join(args.outdir, filename[:filename.rfind('.')] + '_video_depth')
        os.makedirs(output_path, exist_ok=True)
        
        count = 0
        while raw_video.isOpened():
            ret, raw_frame = raw_video.read()
            if not ret:
                break
            
            frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB) / 255.0
            
            frame = transform({'image': frame})['image']
            frame = torch.from_numpy(frame).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                depth = depth_anything(frame)

            depth = F.interpolate(depth[None], (frame_height, frame_width), mode='bilinear', align_corners=False)[0, 0]

            # save depth as binary file
            depth = depth.cpu().numpy()
            print(count)
            np.save(os.path.join(output_path, f"{count:04d}.npy"), depth)
            
            # split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
            # combined_frame = cv2.hconcat([raw_frame, split_region, depth_color])
            # 
            # out.write(combined_frame)
            # cv2.imwrite(os.path.join(output_path, f"{count:04d}.png"), combined_frame)
            cv2.imwrite(os.path.join(output_path, f"{count:04d}.png"), raw_frame)
            count += 1

        raw_video.release()
        # out.release()
