from torchvision.datasets import ImageFolder

class ImageFolderWithPaths(ImageFolder):
    """Modifies torchviison ImageFolder to return (img, label, img_path)"""

    def __getitem__(self, index):

        img, label = super(ImageFolderWithPaths, self).__getitem__(index)

        path = self.imgs[index][0]

        return (img, label, path)
