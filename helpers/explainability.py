from captum.attr import IntegratedGradients, Occlusion
from captum.attr import visualization as viz

from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.pyplot import figure, close, show
import numpy as np

from helpers.visualise import norm_and_transpose_input_tensor

BLUE_CMAP = LinearSegmentedColormap.from_list('custom blue',
                                                 [(0, (252/255,56/255,2/255,1)),
                                                  (0.5, (255/255,255/255,255/255,0)),
                                                  (1, (0,0,255/255,1))], 
                                                 # [(0, (255/255,255/255,255/255,0)),
                                                 #  (0.5, (0,0,255/255,1)),
                                                 #  (1, (160/255,10/255,206/255,1))], 
                                              N=256)

ORED_CMAP = LinearSegmentedColormap.from_list('custom ored',
                                                 [(0, (255/255,255/255,255/255,0)),
                                                  (0.5, (252/255,56/255,2/255,1)),
                                                  (1, (249/255,0/255,149/255,1))], N=256)

GREEN_CMAP = LinearSegmentedColormap.from_list('custom green',
                                                 [(0, (255/255,255/255,255/255,0)),
                                                  (0.5,(2/255,252/255,52/255,1) ),
                                                  (1, (5/255,54/255,252/255,1))], N=256)


def make_int_grad_model(model):
	integrated_gradients = IntegratedGradients(model)
	return integrated_gradients


def make_occ_int_grad_model(model):

	occlusion_model = Occlusion(model)
	return occlusion_model


def get_occ_int_grad_for_single_tensor(model, input_tensor, pred_label_idx, single=False, baseline=0,method='gausslegendre', window=15):
    if str(type(model)) == \
    "<class 'captum.attr._core.integrated_gradients.IntegratedGradients'>":
        attrs = model.attribute(
            input_tensor.unsqueeze(0), 
            target=pred_label_idx, 
            n_steps=40, 
            baselines=baseline,
            method=method
        )
    else:
        channels = 1 if single else 3
        attrs = model.attribute(
            input_tensor.unsqueeze(0), 
            target=pred_label_idx, 
            strides= (channels, -(-window // 2), -(-window // 2)) if window <15 else(channels, 8, 8), 
            sliding_window_shapes=(channels, window, window),
            baselines=baseline)
    
    return attrs


def viz_intgrad_with_bbox(img_attr, input_img_tensor, bbox, title,
    sign='positive', method='blended_heat_map', use_blue=False,
    plt_fig_axis=None):
    """
    Helper function to generate CXR plot with int grad attributions and BBox.

    Args:
    - img_attr (torch.float Tensor): Torch Tensor of attribution.
    - input_img_tensor (torch.float Tensor): The input image

    """
    if use_blue or sign=="all":
        curr_cmap = BLUE_CMAP
    elif sign=="positive":
        curr_cmap = GREEN_CMAP
    else:
        curr_cmap = ORED_CMAP
    
    _ = viz.visualize_image_attr(attr=np.transpose(img_attr.squeeze().cpu().detach().numpy(), (1,2,0)),
                            original_image=norm_and_transpose_input_tensor(input_img_tensor),
                             method=method,
                             cmap=curr_cmap,
                             show_colorbar=True,
                             sign=sign,
                             title=title,
                             use_pyplot=False,
                             plt_fig_axis = plt_fig_axis)
    _[1].add_patch(Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],edgecolor='red',
                         facecolor='none',lw=3))
    return _


def viz_intgrad(img_attr, input_img_tensor, title, sign='positive', method='blended_heat_map',
	use_blue=False, plt_fig_axis=None):
    """Helper function to generate CXR plot with int grad attributions."""
    if use_blue or sign=="all":
        curr_cmap = BLUE_CMAP
    elif sign=="positive":
        curr_cmap = GREEN_CMAP
    else:
        curr_cmap = ORED_CMAP
    
    _ = viz.visualize_image_attr(attr=np.transpose(img_attr.squeeze().cpu().detach().numpy(), (1,2,0)),
                            original_image=norm_and_transpose_input_tensor(input_img_tensor),
                             method=method,
                             cmap=curr_cmap,
                             # cmap=default_cmap,
                             show_colorbar=True,
                             sign=sign,
                             title=title,
                             use_pyplot=False,
                             plt_fig_axis=plt_fig_axis)
    return _

def plot_ig_and_mask(attrs, img_tensor, bbox=None,
                     methods = ['blended_heat_map','blended_heat_map','blended_heat_map','masked_image'],
                     signs = ['positive','positive','negative','positive'], 
                     titles = ["Integrated Gradients", "Positive Attribution (Masked)", "Negative Attribution (Masked)", "Positive Mask"], 
                     figsize = (18, 6),
                     single = False):
    plt_fig = figure(figsize=figsize)
    plt_axis_np = plt_fig.subplots(1, len(attrs), squeeze=True)
    for i in range(0,len(attrs)):
        if single:
            img_attr = attrs[i].expand(3, *attrs[i].shape[1:])
            input_img_tensor = img_tensor.expand(3, *img_tensor.shape[1:])
        else:
            img_attr = attrs[i]
            input_img_tensor = img_tensor
        if bbox:
            viz_intgrad_with_bbox(
                img_attr=img_attr, 
                input_img_tensor= input_img_tensor, 
                bbox = bbox,
                title = titles[i],
                method = methods[i],
                sign = signs[i],
                plt_fig_axis=(plt_fig, plt_axis_np[i]))
        else:
            viz_intgrad(
		        img_attr=attrs[i], 
		        input_img_tensor= img_tensor, 
		        title = titles[i],
		        method = methods[i],
		        sign = signs[i],
                plt_fig_axis=(plt_fig, plt_axis_np[i]))
    plt_fig.tight_layout()
    show()
    close(plt_fig)
