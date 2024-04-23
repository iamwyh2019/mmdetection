import cv2
import numpy as np

from mmdet.apis import inference_detector, init_detector
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

from typing import List, Dict, Callable, Any, Union, Tuple
from mmdet.structures import DetDataSample
import os
import torch

model_name = 'rtmdet-ins_l_8xb32-300e_coco'

config_path = os.path.join(os.path.dirname(__file__), f'configs/rtmdet/{model_name}.py')
checkpoint_path = os.path.join(os.path.dirname(__file__), f'checkpoints/{model_name}.pth')
device = torch.device('cuda:0')

# build the model from a config file and a checkpoint file
model = init_detector(config_path, checkpoint_path, device=device)

CLASSES = model.dataset_meta['classes']
COLORS = model.dataset_meta['palette']

executor = ThreadPoolExecutor(max_workers = 30)

def get_geometric_center(masks: torch.Tensor) -> List[List[int]]:
    N, H, W = masks.shape

    x = torch.arange(W, device=masks.device).view(1, 1, W).expand(N, H, W)
    y = torch.arange(H, device=masks.device).view(1, H, 1).expand(N, H, W)

    x = (x * masks).sum(dim=(1, 2)) / masks.sum(dim=(1, 2))
    y = (y * masks).sum(dim=(1, 2)) / masks.sum(dim=(1, 2))

    return torch.stack([x, y], dim=1).cpu().numpy().astype(int).tolist()

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

    # get the geometric centers
    centers = get_geometric_center(pred_instances.masks)

    # convert to numpy
    boxes: np.ndarray = pred_instances.bboxes.to(torch.int32).cpu().numpy()
    labels: np.ndarray = pred_instances.labels.cpu().numpy()
    scores: np.ndarray = pred_instances.scores.cpu().numpy()
    masks: np.ndarray = pred_instances.masks.to(torch.uint8).cpu().numpy()

    return masks, boxes, labels, scores, centers


count = 0
total_used_time = 0

def process_image(image:np.ndarray,
                  score_threshold: float = 0.3,
                  top_k: int = 15) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, np.ndarray, np.ndarray, List[List[int]]]:
    global model
    global count, total_used_time

    start_time = time.time()
    result = inference_detector(model, image)
    used_time = time.time() - start_time
    total_used_time += used_time
    count += 1

    if count > 30:
        latency = total_used_time / count
        print(f'latency: {latency:.3f}s ({1/latency:.3f} fps)')
        count = 0
        total_used_time = 0

    masks, boxes, labels, scores, geometry_center = parse_result(result, score_threshold, top_k)

    # get contours and geometry centers
    mask_contours = [None for _ in range(len(masks))]
    for i, mask in enumerate(masks):
        # crop the mask by box plus padding of 5 pixels
        x1, y1, x2, y2 = boxes[i]
        # x1 = max(0, x1 - 5)
        # y1 = max(0, y1 - 5)
        # x2 = min(image.shape[1], x2 + 5)
        # y2 = min(image.shape[0], y2 + 5)
        mask_crop = mask[y1:y2, x1:x2]

        if mask_crop.size == 0:
            print('mask_crop size is 0', i, CLASSES[labels[i]])
            print('boxes[i]', boxes[i])
            print('mask.shape', mask.shape)
            print('mask_crop.shape', mask_crop.shape)
            cv2.imwrite('bad_image.png', image)
        if mask_crop.max() == 0:
            nonzero = np.nonzero(mask)
            if len(nonzero) == 0:
                print('no nonzero', i, CLASSES[labels[i]])
            x1 = min(nonzero[1])
            x2 = max(nonzero[1])
            y1 = min(nonzero[0])
            y2 = max(nonzero[0])
            mask_crop = mask[y1:y2, x1:x2]
            boxes[i] = [x1, y1, x2, y2]

        contours = bitmap_to_polygon(mask_crop)

        # find the largest contour
        contours.sort(key=lambda x: len(x), reverse=True)
        largest_contour = max(contours, key = cv2.contourArea)

        # convert the contour to be relative to the top-left corner of the box
        # largest_contour = largest_contour + np.array([x1, y1])
        # largest_contour = largest_contour - np.array([boxes[i][0], boxes[i][1]])

        mask_contours[i] = largest_contour

    return masks, mask_contours, boxes, labels, scores, geometry_center


def get_recognition(image: np.ndarray,
                    filter_objects: List[str] = [],
                    score_threshold: float = 0.3,
                    top_k: int = 15) -> Dict[str, Any]:
    global CLASSES, COLORS

    masks, mask_contours, boxes, labels, scores, geometry_center = process_image(
        image,
        score_threshold,
        top_k,
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
        result = get_filtered_objects(result, filter_objects)

    return result


async def async_get_recognition(image: np.ndarray,
                                filter_objects: List[str] = [],
                                score_threshold: float = 0.3,
                                top_k: int = 15) -> Dict[str, Any]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, get_recognition, image, filter_objects, score_threshold, top_k)


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
            # contour is relative to the box, need to add the box's top-left corner
            x1, y1, _, _ = boxes[i]
            contour = np.array(contour) + np.array([x1, y1])
            contour = np.array(contour, dtype=np.int32)
            color = color_list[i]
            cv2.drawContours(image, [contour], -1, color, 1)

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
    outs = cv2.findContours(bitmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = outs[-2]
    contours = [c.reshape(-1, 2) for c in contours]
    return contours


def main():
    capture = cv2.VideoCapture(0)
    # set resolution to 640x360
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    print("Resolution:", capture.get(cv2.CAP_PROP_FRAME_WIDTH), "x", capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    count = 0
    total_used_time = 0

    while True:
        ret, image = capture.read()
        if not ret:
            break

        # plot image
        cv2.imshow('image', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        start_time = time.time()
        result = get_recognition(image, score_threshold=0.5)
        used_time = time.time() - start_time
        total_used_time += used_time
        count += 1

        if count > 30:
            latency = total_used_time / count
            print(f'latency: {latency:.3f}s ({1/latency:.3f} fps)')
            count = 0
            total_used_time = 0

    # image = async_draw_recognition(image, result, draw_contour=True, draw_text=True, draw_score=True)
    # cv2.imshow('result', image)
    # cv2.waitKey(0)

    # bad_image = cv2.imread('bad_image.png')
    # print(bad_image.shape, bad_image.dtype)

    # result = inference_detector(model, bad_image)
    # ins = result.pred_instances
    # ins = ins[ins.scores >= 0.3]
    # boxes: np.ndarray = ins.bboxes.to(torch.int32).cpu().numpy()
    # labels: np.ndarray = ins.labels.cpu().numpy()
    # scores: np.ndarray = ins.scores.cpu().numpy()
    # masks: np.ndarray = ins.masks.to(torch.uint8).cpu().numpy()

    # for i, mask in enumerate(masks):
    #     print(i, CLASSES[labels[i]], mask.shape, boxes[i], scores[i])

if __name__ == '__main__':
    main()