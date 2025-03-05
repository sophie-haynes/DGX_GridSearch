from numpy import asarray


def norm_and_transpose_input_tensor(curr_img_tensor):
    """Normalises and transposes an input image tensor for viz"""
    return asarray((curr_img_tensor.cpu().numpy() -
                    curr_img_tensor.cpu().numpy().min()) /
                   (curr_img_tensor.cpu().numpy().max() -
                    curr_img_tensor.cpu().cpu().numpy()
                    .min())).transpose(1, 2, 0)

def norm_input_tensor(curr_img_tensor):
    """Normalises input image tensor for viz"""
    return asarray((curr_img_tensor.cpu().numpy() -
                    curr_img_tensor.cpu().numpy().min()) /
                   (curr_img_tensor.cpu().numpy().max() -
                    curr_img_tensor.cpu().cpu().numpy()
                    .min()))