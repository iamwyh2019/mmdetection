import cv2
import numpy as np
from mmengine.utils import track_iter_progress

from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS

import asyncio
from concurrent.futures import ThreadPoolExecutor

from typing import List, Dict, Callable, Any, Union, Tuple
from mmdet.structures import DetDataSample

config_path = 'configs/rtmdet/rtmdet-ins_tiny_8xb32-300e_coco.py'
checkpoint_path = 'checkpoints/rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth'
device = 'cuda:0'

# build the model from a config file and a checkpoint file
model = init_detector(config_path, checkpoint_path, device=device)

CLASSES = model.dataset_meta['classes']
COLORS = model.dataset_meta['palette']

executor = ThreadPoolExecutor(max_workers = 30)

def parse_result(result: DetDataSample,
                 score_threshold: float = 0.15,
                 top_k: int = 15) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse the detection results."""
    pred_instances = result.pred_instances

    # first, filter out the low score boxes
    valid_inds = pred_instances.scores > score_threshold
    pred_instances = pred_instances[valid_inds]

    # sort the scores
    scores = pred_instances.scores
    _, sort_inds = scores.sort(descending=True)
    pred_instances = pred_instances[sort_inds]

    # get the top_k
    pred_instances = pred_instances[:top_k]

    # convert to numpy
    bboxes: np.ndarray = pred_instances.bboxes.cpu().numpy()
    labels: np.ndarray = pred_instances.labels.cpu().numpy()
    scores: np.ndarray = pred_instances.scores.cpu().numpy()
    masks: np.ndarray = pred_instances.masks.cpu().numpy()

    return masks, bboxes, labels, scores


def process_image(image:np.ndarray,
                  score_threshold: float = 0.3,
                  top_k: int = 15) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, np.ndarray, np.ndarray, List[List[int]]]:
    global model

    # inference
    result = inference_detector(model, image)
    masks, bboxes, labels, scores = parse_result(result, score_threshold, top_k)
    masks = masks.astype(np.uint8)

    # get contours and geometry centers
    mask_contours = []
    geometry_center = []
    for mask in masks:
        contour, _ = bitmap_to_polygon(mask)
        mask_contours.append(contour)

        # mask_crop is 0/1 matrix
        # the center is the center of mass (assume uniform density)
        # the contour is the largest contour
        center = cv2.moments(mask)
        center_x = int(center["m10"] / center["m00"])
        center_y = int(center["m01"] / center["m00"])
        geometry_center.append([center_x.item(), center_y.item()])

    return masks, mask_contours, bboxes, labels, scores, geometry_center


# for frame in track_iter_progress((video_reader, len(video_reader)), task_name="frame"):
#     result = inference_detector(model, frame, test_pipeline=test_pipeline)
#     visualizer.add_datasample(
#         name='video',
#         image=frame,
#         data_sample=result,
#         draw_gt=False,
#         show=False,
#         pred_score_thr=args.score_thr)
#     frame = visualizer.get_image()


def bitmap_to_polygon(bitmap):
    """Convert masks from the form of bitmaps to polygons.

    Args:
        bitmap (ndarray): masks in bitmap representation.

    Return:
        list[ndarray]: the converted mask in polygon representation.
        bool: whether the mask has holes.
    """
    # cv2.RETR_CCOMP: retrieves all of the contours and organizes them
    #   into a two-level hierarchy. At the top level, there are external
    #   boundaries of the components. At the second level, there are
    #   boundaries of the holes. If there is another contour inside a hole
    #   of a connected component, it is still put at the top level.
    # cv2.CHAIN_APPROX_NONE: stores absolutely all the contour points.
    outs = cv2.findContours(bitmap, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = outs[-2]
    hierarchy = outs[-1]
    if hierarchy is None:
        return [], False
    # hierarchy[i]: 4 elements, for the indexes of next, previous,
    # parent, or nested contours. If there is no corresponding contour,
    # it will be -1.
    with_hole = (hierarchy.reshape(-1, 4)[:, 3] >= 0).any()
    contours = [c.reshape(-1, 2) for c in contours]
    return contours, with_hole