import numpy as np

def calculate_iou(true_bbox, pred_bbox):
    """
    Calculate the iou between the true and predicted bounding boxes.

    Args:
    - true_bbox (array, [x1,y1,x2,y2]): 
    - pred_bbox (array, [x1,y1,x2,y2]): 

    Returns:
    - iou (float)
    
    """
    x1_true, y1_true, x2_true, y2_true = true_bbox
    x1_pred, y1_pred, x2_pred, y2_pred = pred_bbox
    
    intersect_x1 = np.max([x1_true, x1_pred])
    intersect_x2 = np.min([x2_true, x2_pred])
    intersect_y1 = np.max([y1_true, y1_pred])
    intersect_y2 = np.min([y2_true, y2_pred])
    
    intersect_height = np.max([intersect_y2-intersect_y1, np.array(0.)])
    intersect_width = np.max([intersect_x2-intersect_x1, np.array(0.)])
    intersect_area = intersect_height*intersect_width

    true_height = true_bbox[3] - true_bbox[1]+1
    true_width =  true_bbox[2] - true_bbox[0]+1
    pred_height = pred_bbox[3] - pred_bbox[1]+1
    pred_width =  pred_bbox[2] - pred_bbox[0]+1

    union_area = (true_height*true_width) + (pred_height * pred_width) - intersect_area
    return intersect_area / union_area

def calculate_intersect_coverage(true_bbox, pred_bbox):
    x1_true, y1_true, x2_true, y2_true = true_bbox
    x1_pred, y1_pred, x2_pred, y2_pred = pred_bbox
    
    true_height = true_bbox[3] - true_bbox[1]+1
    true_width =  true_bbox[2] - true_bbox[0]+1
    true_area = true_height*true_width
    
    intersect_x1 = np.max([x1_true, x1_pred])
    intersect_x2 = np.min([x2_true, x2_pred])
    intersect_y1 = np.max([y1_true, y1_pred])
    intersect_y2 = np.min([y2_true, y2_pred])
    
    intersect_height = np.max([intersect_y2-intersect_y1, np.array(0.)])
    intersect_width = np.max([intersect_x2-intersect_x1, np.array(0.)])
    intersect_area = intersect_height*intersect_width
    return (intersect_area/true_area)

def calculate_bbox_metrics(true_bbox, pred_bboxes, prediction=None):
    best_iou = 0
    best_bbox = None
    for pred_bbox in pred_bboxes:
        current_iou = calculate_iou(true_bbox, pred_bbox)
        if current_iou > best_iou:
            best_iou = current_iou
            best_bbox = pred_bbox
        