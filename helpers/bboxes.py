import csv
from torch import zeros_like

def resize_bboxes(bboxes, original_size, target_size=512):
    """
    Resize bounding boxes to match a new image size.

    Args:
    - bboxes (list of lists): List of bounding boxes [x, y, width, height]
    - original_size (int): Original size of the image (e.g., 224, 1024)
    - target_size (int): Target size of the image (default is 512)

    Returns:
    - List of resized bounding boxes
    """
    scale = target_size / original_size
    resized_bboxes = [[x * scale, y * scale, w * scale,
                       h * scale] for x, y, w, h in bboxes]
    return resized_bboxes


def read_bounding_boxes(bbox_csv_path):
    """
    Read the bounding box CSV file and return a dictionary with 
    image names as keys and bounding boxes as values.
    """
    bbox_dict = {}

    with open(bbox_csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            img_name = row['img_name']
            # Convert string to a list
            # (e.g., "[32, 119, 15, 13]" -> [32, 119, 15, 13])
            bbox_224 = eval(row['bbox_224'])
            # Store the bounding box (x, y, width, height)
            bbox_dict[img_name] = bbox_224

    return bbox_dict

def calculate_cam_iou(cam_mask, bounding_boxes):
    """Calculate IoU between CAM mask and ground-truth bounding boxes."""
    # cam_mask = (cam_mask > 0.5).float()  # Binary mask
    ious = []
    #range(cam_mask.size(0)):
    for i in range(cam_mask.shape[0]):
        # Get ground-truth box (xmin, ymin, xmax, ymax)
        box = bounding_boxes[i]  
        cam_region = cam_mask[i].nonzero()  # Get CAM region
        
        if cam_region.size(0) == 0:  # If no region, IoU is 0
            ious.append(0.0)
            continue
        
        # Create a mask for the bounding box
        box_mask = zeros_like(cam_mask[i])
        box_mask[box[1]:box[3], box[0]:box[2]] = 1.0

        # Calculate intersection and union between CAM region and 
        # ground-truth box
        intersection = (cam_mask[i] * box_mask).sum()
        union = (cam_mask[i] + box_mask).clamp(0, 1).sum()

        iou = intersection / union
        ious.append(iou.item())

    return ious

