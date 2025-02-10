import cv2
from helpers.explainability import get_occ_int_grad_for_single_tensor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

def convert_to_uint8(image):
    uint8_image = image-image.min()
    uint8_image = uint8_image / uint8_image.max() * 255
    uint8_image = uint8_image.astype(np.uint8)
    return uint8_image

def get_attrs(image, label, model, int_model=None, occ_model=None, device="cpu", single=False, window=8):
    """
    Get occlusion and integrated gradient attributions for a given image.

    Args:
    - image (torch.float Tensor): The input image.
    - label (int): Index of image label (0: "Normal", 1: "Nodule").
    - model (torchvision model): Inference model to obtain attributions from. 
    - int_model (optional, Captum IntegratedGradients model): Int Grad model based on inference model.
    - occ_model (optional, Captum Occlusion model): Occlusion model based on inference model.
    - device (torch.device): Device to perform caluclations on (CUDA/CPU).
    - single (boolean): Indicate whether image and model are single channel.
    - window (int): Window size for occlusion window.

    Returns:
    - int_attr
    - occ_attr
    """
    
    model=model.to(device)
    int_attrs = get_occ_int_grad_for_single_tensor(int_model, image, label, single=single, window=window)
    occ_attrs = get_occ_int_grad_for_single_tensor(occ_model, image, label, single=single, window=window)
    model = model.to("cpu")
    cv2_occ_attr = cv2.cvtColor(occ_attrs[0][0].unsqueeze(2).cpu().numpy(), cv2.COLOR_RGB2BGR)
    cv2_int_attr = int_attrs[0][0].unsqueeze(2).cpu().numpy()
    return cv2_int_attr[0][0].unsqueeze(2).cpu().numpy(), cv2_occ_attr[0][0].unsqueeze(2).cpu().numpy()
    

def threshold_attributions(attr, min_thresh_frac=0.15, non_zero=False, binary_mask=False):
    if binary_mask:
        mask_type = cv2.THRESH_BINARY
    else:
        mask_type = cv2.THRESH_TOZERO
    
    thresh_attr = cv2.threshold(src = attr,
                                thresh = 0 if non_zero else attr.max()*min_thresh_frac, 
                                maxval = attr.max(),
                                type = mask_type)[1]
    return thresh_attr

def get_area_thresheld_connected_components(mask, size_thresh = 1, img=None):
    """
    Returns connected components from mask.

    Args:
    - mask (): Single channel attributtion mask
    - size_thresh (int, optional): minimum area for connected component

    Returns:
    - df (DataFrame): ["centroid","xmin","xmax","ymin","ymax"]
    - centroids (2D Numpy array): array of [x,y] pairs
    
    """
    df = pd.DataFrame(columns=["centroid","xmin","xmax","ymin","ymax"])
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(convert_to_uint8(mask))
    if img is not None:
        copy_img = convert_to_uint8(img.copy())
    # save idx of valid components
    size_thresh_idx = []
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= size_thresh:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            size_thresh_idx.append(i)
            # Add a new row using loc[]
            df.loc[len(df)] = [centroids[i], x, x+w, y-h, y]
            if img is not None:
                cv2.rectangle(copy_img, (x, y), (x+w, y+h), (0, 255, 0), thickness=1)
                cv2.circle(copy_img, centroids[i].astype(int), 1, (255,0,0), thickness=1)
    if img is not None:
        plt.imshow(copy_img)
        plt.axis("off")
        
    return df


def attribution_bbox_with_connected_components(mask, size_thresh = 1, img=None):
    """
    Returns connected components from mask.

    Args:
    - mask (): Single channel attributtion mask
    - size_thresh (int, optional): minimum area for connected component

    Returns:
    - df (DataFrame): ["centroid","xmin","xmax","ymin","ymax"]
    - centroids (2D Numpy array): array of [x,y] pairs
    
    """
    df = pd.DataFrame(columns=["centroid","xmin","xmax","ymin","ymax"])
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(convert_to_uint8(mask))
    if img is not None:
        copy_img = convert_to_uint8(img.copy())
    
    # save idx of valid components
    size_thresh_idx = []
    
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= size_thresh:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            size_thresh_idx.append(i)
            # Add a new row using loc[]
            df.loc[len(df)] = [centroids[i], x, x+w, y-h, y]
            if img is not None:
                cv2.rectangle(copy_img, (x, y), (x+w, y+h), (0, 255, 0), thickness=1)
                cv2.circle(copy_img, centroids[i].astype(int), 1, (255,0,0), thickness=1)
    if img is not None:
        plt.imshow(copy_img)
        plt.axis("off")
        
    return df
        
    
def plt_area_thresheld_connected_components(mask, plot_image, size_thresh = 1):
    """Visualise the connected componenents and centroids"""
    # duplicate image to prevent overwriting base
    copy_img = convert_to_uint8(plot_image.copy())
    # generate cc
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(convert_to_uint8(mask))
    
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= size_thresh:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            # plot cc bbox
            cv2.rectangle(copy_img, (x, y), (x+w, y+h), (0, 255, 0), thickness=1)
            # plot cc centroid
            cv2.circle(copy_img, centroids[i].astype(int), 1, (255,0,0), thickness=1)
    plt.imshow(copy_img)
    plt.axis("off")


def get_area_thresheld_connected_components(mask, size_thresh = 1, img=None):
    """
    Returns connected components from mask.

    Args:
    - mask (): Single channel attributtion mask
    - size_thresh (int, optional): minimum area for connected component

    Returns:
    - df (DataFrame): ["centroid","xmin","xmax","ymin","ymax"]
    - centroids (2D Numpy array): array of [x,y] pairs
    
    """
    df = pd.DataFrame(columns=["centroid","xmin","xmax","ymin","ymax"])
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(convert_to_uint8(mask))
    if img is not None:
        copy_img = convert_to_uint8(img.copy())
    # save idx of valid components
    size_thresh_idx = []
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= size_thresh:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            size_thresh_idx.append(i)
            # Add a new row using loc[]
            df.loc[len(df)] = [centroids[i], x, x+w, y-h, y]
            if img is not None:
                cv2.rectangle(copy_img, (x, y), (x+w, y+h), (0, 255, 0), thickness=1)
                cv2.circle(copy_img, centroids[i].astype(int), 1, (255,0,0), thickness=1)
    if img is not None:
        plt.imshow(copy_img)
        plt.axis("off")
        
    return df
    # return centroids[size_thresh_idx].astype(int)

def df_cc_bboxes_by_dbscan(df, max_distance=10, min_samples=1, margin=0, x_shape=512, y_shape=512, thresh_area = 4):
    """Generate b"""
    # get centroids from df and convert into correct format
    centroids = np.array(df['centroid'].values.tolist())
    # apply DBSCAN to centroids
    db = DBSCAN(eps=max_distance, min_samples=min_samples).fit(centroids)
    # get group labels from DBSCAN
    labels = db.labels_
    
    # Calculate bounding boxes for each cluster
    grouped_bboxes = []
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        # cluster_points = centroids[labels == label]
        cluster_df = df[labels==label]
        
        # Calculate the bounding box for the cluster
        # x_min = np.min(cluster_points[:, 0])
        # y_min = np.min(cluster_points[:, 1])
        # x_max = np.max(cluster_points[:, 0])
        # y_max = np.max(cluster_points[:, 1])

        x_min = cluster_df.xmin.min()
        x_max = cluster_df.xmax.max()
        y_min = cluster_df.ymin.min()
        y_max = cluster_df.ymax.max()
        
        # Apply margin to expand the bounding box
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(x_shape - 1, x_max + margin)
        y_max = min(y_shape - 1, y_max + margin)

        if x_max-x_min > 0 and y_max-y_min > 0 and ((x_max-x_min)*(y_max-y_min)>=thresh_area):
            grouped_bboxes.append((x_min, y_min, x_max, y_max))
    
    return grouped_bboxes
    
