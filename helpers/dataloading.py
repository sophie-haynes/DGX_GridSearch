from torchvision.datasets import ImageFolder

class ImageFolderWithPaths(ImageFolder):
    """Modifies torchviison ImageFolder to return (img, label, img_path)"""

    def __getitem__(self, index):

        img, label = super(ImageFolderWithPaths, self).__getitem__(index)

        path = self.imgs[index][0]

        return (img, label, path)

def remap_labels(label):
    """Helper function to map class labels ['1_nodule', '0_normal'] to indices. """
    mapping_dict = {trainset.class_to_idx['1_nodule']: 1, trainset.class_to_idx['0_normal']: 0}
    return mapping_dict[label]