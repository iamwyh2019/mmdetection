import cv2
import numpy as np

from mmdet.apis import inference_detector, init_detector

import asyncio
from multiprocessing import Process, Queue

from typing import List, Dict, Callable, Any, Union, Tuple
from mmdet.structures import DetDataSample
import os
import torch

METAINFO = {
        'classes':
        ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
         'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
         'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
         'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
         'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
         'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
         'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
         'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
         'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
         'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
         'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
         'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
         'scissors', 'teddy bear', 'hair drier', 'toothbrush'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
         (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
         (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
         (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
         (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
         (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
         (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
         (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
         (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),
         (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51),
         (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
         (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
         (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
         (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
         (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
         (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
         (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
         (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
         (246, 0, 122), (191, 162, 208)]
    }

CLASSES = METAINFO['classes']
COLORS = METAINFO['palette']

def model_process(input_queue: Queue, output_queue: Queue, config: str, checkpoint: str, device: str = 'cuda:0'):
    device = torch.device(device)
    model = init_detector(config, checkpoint, device=device)
    while True:
        image = input_queue.get()
        if image is None:
            break
        result = inference_detector(model, image)
        output_queue.put(result)

def async_model_setup(config: str, checkpoint: str, device: str = 'cuda:0') -> Tuple[Process, Queue, Queue]:
    input_queue = Queue()
    output_queue = Queue()
    model_proc = Process(target=model_process, args=(input_queue, output_queue, config, checkpoint, device))
    model_proc.start()
    return model_proc, input_queue, output_queue

model_name = 'rtmdet-ins_l_8xb32-300e_coco'

config_path = os.path.join(os.path.dirname(__file__), f'configs/rtmdet/{model_name}.py')
checkpoint_path = os.path.join(os.path.dirname(__file__), f'checkpoints/{model_name}.pth')
device = 'cuda:0'

input_queue = None
output_queue = None
model_proc = None

def init_model() -> None:
    global model_proc, input_queue, output_queue
    model_proc, input_queue, output_queue = async_model_setup(config_path, checkpoint_path, device)

def close_model() -> None:
    input_queue.put(None)
    model_proc.join()
    input_queue.close()
    output_queue.close()

def parse_result(result: DetDataSample,
                 score_threshold: float = 0.15,
                 top_k: int = 15) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse the detection results."""
    pred_instances = result.pred_instances

    # first, filter out the low score boxes
    valid_inds = pred_instances.scores >= score_threshold
    pred_instances = pred_instances[valid_inds]

    # sort the scores
    scores = pred_instances.scores
    _, sort_inds = scores.sort(descending=True)
    pred_instances = pred_instances[sort_inds]

    # get the top_k
    pred_instances = pred_instances[:top_k]

    # convert to numpy
    boxes: np.ndarray = pred_instances.bboxes.cpu().numpy()
    labels: np.ndarray = pred_instances.labels.cpu().numpy()
    scores: np.ndarray = pred_instances.scores.cpu().numpy()
    masks: np.ndarray = pred_instances.masks.cpu().numpy()

    # boxes need to be rounded
    boxes = np.round(boxes).astype(np.int32)

    return masks, boxes, labels, scores


async def process_image(image:np.ndarray,
                  score_threshold: float = 0.3,
                  top_k: int = 15,
                  stats = None) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, np.ndarray, np.ndarray, List[List[int]]]:

    # inference
    if stats:
        stats.start_event('inference')

    input_queue.put(image)

    result = await asyncio.get_event_loop().run_in_executor(None, output_queue.get)

    if stats:
        stats.end_event('inference')
        stats.start_event('parse')

    masks, boxes, labels, scores = parse_result(result, score_threshold, top_k)
    masks = masks.astype(np.uint8)

    if stats:
        stats.end_event('parse')
        stats.start_event('draw_contour')

    # get contours and geometry centers
    mask_contours = []
    geometry_center = []
    for mask in masks:
        contours, hole = bitmap_to_polygon(mask)
        # find the largest contour
        contours.sort(key=lambda x: len(x), reverse=True)
        largest_contour = max(contours, key = cv2.contourArea)
        mask_contours.append(largest_contour)

        # mask_crop is 0/1 matrix
        # the center is the center of mass (assume uniform density)
        # the contour is the largest contour
        center = cv2.moments(mask)
        center_x = int(center["m10"] / center["m00"])
        center_y = int(center["m01"] / center["m00"])
        geometry_center.append([center_x, center_y])

        if stats:
            stats.end_event('draw_contour')

    # # crop the mask by box
    # for i, box in enumerate(boxes):
    #     x1, y1, x2, y2 = box
    #     masks[i] = masks[i][y1:y2, x1:x2]
    return masks, mask_contours, boxes, labels, scores, geometry_center


async def async_get_recognition(image: np.ndarray,
                    filter_objects: List[str] = [],
                    score_threshold: float = 0.3,
                    top_k: int = 15,
                    stats = None) -> Dict[str, Any]:
    
    masks, mask_contours, boxes, labels, scores, geometry_center = await process_image(
        image,
        score_threshold,
        top_k,
        stats
    )

    mask_contours = [contour.tolist() for contour in mask_contours]
    boxes = boxes.tolist()
    scores = scores.tolist()
    labels = labels.tolist()

    # convert labels to class names
    class_names = [CLASSES[label] for label in labels]
    
    # get colors
    # each color is a 3-tuple (R, G, B)
    color_list = [COLORS[label] for label in labels]

    result = {
        "masks": masks,
        "mask_contours": mask_contours,
        "boxes": boxes,
        "scores": scores,
        "labels": labels, # "labels" is the original label, "class_names" is the class name
        "class_names": class_names,
        "geometry_center": geometry_center,
        "colors": color_list,
    }

    if filter_objects:
        if stats:
            stats.start_event('filter')
        # filter the objects
        result = get_filtered_objects(result, filter_objects)
        if stats:
            stats.end_event('filter')

    return result


def async_draw_recognition(image: np.ndarray, result: Dict[str, Any],
                     black: bool = False, draw_contour: bool = False, draw_mask: bool = True, 
                     draw_box: bool = False, draw_text: bool = True, draw_score = True,
                     draw_center = False,
                     alpha: float = 0.45) -> np.ndarray:
    masks = result['masks']
    mask_contours = result['mask_contours']
    boxes = result['boxes']
    class_names = result['class_names']
    scores = result['scores']
    geometry_center = result['geometry_center']
    labels = result['labels']
    
    if black:
        image = np.zeros_like(image)

    if len(masks) == 0:
        return image

    # colors
    # each color is a 3-tuple (B, G, R)
    colors = []
    color_list = []
    for label in labels:
        color = COLORS[label]
        color = (color[2], color[1], color[0])
        color_list.append(color)
        colors.append(np.array(color, dtype=float).reshape(1,1,1,3))
    colors = np.concatenate(colors, axis=0)

    # yield to other tasks
    # await asyncio.sleep(0)
    
    if draw_mask:
        # masks N*H*W
        masks = np.array(masks, dtype=float)
        # change to N*H*W*1
        masks = np.expand_dims(masks, axis=3)

        masks_color = masks.repeat(3, axis=3) * colors * alpha

        inv_alpha_masks = masks * (-alpha) + 1

        masks_color_summand = masks_color[0]
        if len(masks_color) > 1:
            inv_alpha_cumul = inv_alpha_masks[:-1].cumprod(axis=0)
            masks_color_cumul = masks_color[1:] * inv_alpha_cumul
            masks_color_summand += masks_color_cumul.sum(axis=0)

        image = image * inv_alpha_masks.prod(axis=0) + masks_color_summand
        image = image.astype(np.uint8)

    # yield to other tasks
    # await asyncio.sleep(0)

    # draw the contours
    if draw_contour:
        for i, contour in enumerate(mask_contours):
            contour = np.array(contour, dtype=np.int32)
            color = color_list[i]
            cv2.drawContours(image, [contour], -1, color, 2)

    # yield to other tasks
    # await asyncio.sleep(0)

    # draw box
    if draw_box:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            color = color_list[i]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)

    # yield to other tasks
    # await asyncio.sleep(0)

    # place text at the center
    if draw_text:
        for i, center in enumerate(geometry_center):
            text = class_names[i]
            if draw_score:
                text += f' {scores[i]:.2f}'
            cv2.putText(image, text, center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # draw center with a green circle, and center of bounding box with a yellow circle
    if draw_center:
        for i, center in enumerate(geometry_center):
            cv2.circle(image, tuple(center), 3, (0, 255, 0), -1)
            x1, y1, x2, y2 = boxes[i]
            center_box = ((x1+x2)//2, (y1+y2)//2)
            cv2.circle(image, center_box, 3, (0, 255, 255), -1)
    
    return image


def get_filtered_objects(result: Dict[str, Any], filter_objects: List[str]) -> Dict[str, Any]:
    """Filter the objects by class names."""
    assert 'class_names' in result

    fields = result.keys()
    new_result = {field: [] for field in fields}
    class_names = result['class_names']

    for i, class_name in enumerate(class_names):
        if class_name in filter_objects:
            for field in fields:
                new_result[field].append(result[field][i])

    return new_result


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


async def main():
    init_model()

    image = cv2.imread('demo/large_image.jpg')

    result = await async_get_recognition(image)
    image = async_draw_recognition(image, result, draw_contour=True, draw_text=True, draw_score=True)
    cv2.imshow('result', image)
    cv2.waitKey(0)

    close_model()

if __name__ == '__main__':
    asyncio.run(main())