# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import cv2
import mmcv
import torch

from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS

# record time
import time
from collections import deque

latency = {
    'read': deque(maxlen=100),
    'inference': deque(maxlen=100),
    'visualize': deque(maxlen=100),
}


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--camera-id', type=int, default=0, help='camera device id')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # build the model from a config file and a checkpoint file
    device = torch.device(args.device)
    model = init_detector(args.config, args.checkpoint, device=device)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    visualizer.dataset_meta = model.dataset_meta

    camera = cv2.VideoCapture(args.camera_id)
    # set resolution: 760*428
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 760)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 428)
    print("Resolution: {}x{}".format(camera.get(cv2.CAP_PROP_FRAME_WIDTH), camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    print('Press "Esc", "q" or "Q" to exit.')

    last_print = time.time()

    while True:
        start_time = time.time()
        ret_val, img = camera.read()
        latency['read'].append(time.time() - start_time)

        start_time = time.time()
        result = inference_detector(model, img)
        latency['inference'].append(time.time() - start_time)

        # img = mmcv.imconvert(img, 'bgr', 'rgb')
        # visualizer.add_datasample(
        #     name='result',
        #     image=img,
        #     data_sample=result,
        #     draw_gt=False,
        #     pred_score_thr=args.score_thr,
        #     show=False)

        # img = visualizer.get_image()
        # img = mmcv.imconvert(img, 'bgr', 'rgb')
        # cv2.imshow('result', img)

        # ch = cv2.waitKey(1)
        # if ch == 27 or ch == ord('q') or ch == ord('Q'):
        #     break

        # print mean latency of each stage
        if time.time() - last_print > 5.0:
            print('read: {:.3f}s, inference: {:.3f}s'.format(
                sum(latency['read']) / len(latency['read']),
                sum(latency['inference']) / len(latency['inference'])))
            last_print = time.time()


if __name__ == '__main__':
    main()
