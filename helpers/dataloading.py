from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from helpers.cxr import get_cxr_dataset_normalisation

class ImageFolderWithPaths(ImageFolder):
    """Modifies torchviison ImageFolder to return (img, label, img_path)"""

    def __getitem__(self, index):

        img, label = super(ImageFolderWithPaths, self).__getitem__(index)

        path = self.imgs[index][0]

        return (img, label, path)

def load_dataset_with_paths(dataset_path, dataset_name, process="arch", 
    crop_size=512, batch_size=4, shuffle=True):
    """Wrapper helper to load a dataset with img paths."""
    dataset = ImageFolderWithPaths(
        root = dataset_path,
        transform = get_cxr_eval_transforms(
            crop_size = crop_size,
            normalise = get_cxr_dataset_normalisation(
                dataset = dataset_name, 
                process = process
                )
            )
        )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def load_dataset(dataset_path, dataset_name, process="arch", 
    crop_size=512, batch_size=4, shuffle=True):
    """Wrapper helper to load a dataset."""
    dataset = ImageFolder(
        root = dataset_path,
        transform = get_cxr_eval_transforms(
            crop_size = crop_size,
            normalise = get_cxr_dataset_normalisation(
                dataset = dataset_name, 
                process = process
                )
            )
        )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)