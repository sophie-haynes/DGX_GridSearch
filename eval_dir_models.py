import argparse, copy, os
import torch, torchvision
from torcheval.metrics import BinaryAUROC, BinaryF1Score, BinaryPrecision, BinaryRecall
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2


arch_seg_dict = {
    # data       mean        std
    'cxr14': [[128.2716], [76.7148]],
    'openi': [[127.7211], [69.7704]],
    'jsrt': [[139.9666], [72.4017]],
    'padchest': [[129.5006], [72.6308]],
}

def convert_to_single_channel(model):
    """
    Modifies the first convolutional layer of a given model to accept single-channel input.
    
    Args:
        model (torch.nn.Module): The model to be modified.
        
    Returns:
        torch.nn.Module: The modified model with a single-channel input.
    """
    # Identify the first convolutional layer
    conv1 = None
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            conv1 = layer
            conv1_name = name
            break
    
    if conv1 is None:
        raise ValueError("The model does not have a Conv2D layer.")
    
    # Create a new convolutional layer with the same parameters except for the input channels
    new_conv1 = nn.Conv2d(
        in_channels=1,  # Change input channels to 1
        out_channels=conv1.out_channels,
        kernel_size=conv1.kernel_size,
        stride=conv1.stride,
        padding=conv1.padding,
        bias=conv1.bias is not None
    )
    
    # Replace the old conv1 layer with the new one
    def recursive_setattr(model, attr, value):
        attr_list = attr.split('.')
        for attr_name in attr_list[:-1]:
            model = getattr(model, attr_name)
        setattr(model, attr_list[-1], value)
    
    recursive_setattr(model, conv1_name, new_conv1)
    
    return model

def get_cxr_eval_transforms(crop_size,normalise, single=False):
    if single:
        grey = v2.Grayscale(num_output_channels=1)
    else:
        grey = v2.Grayscale(num_output_channels=3)
    cxr_transform_list = [
        v2.ToImage(),
        grey,
        v2.Resize(size=crop_size,antialias=True),
        v2.ToDtype(torch.float32, scale=False),
        normalise
    ]
    return cxr_transform_list

def evaluate_gpu_metrics(model, dataloader, device):
    auroc = BinaryAUROC()
    f1 = BinaryF1Score()
    prec =  BinaryPrecision()
    rec = BinaryRecall()

    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            auroc.update(torch.softmax(outputs, dim=1)[:, 1],labels)
            f1.update(torch.softmax(outputs, dim=1)[:, 1],labels)
            prec.update(torch.softmax(outputs, dim=1)[:, 1],labels)
            rec.update(torch.softmax(outputs, dim=1)[:, 1],labels)
    return prec.compute().item(), rec.compute().item(), f1.compute().item(), auroc.compute().item()

def parse_args():
    parser = argparse.ArgumentParser(description="Test existing models")

    # Add arguments for the grid search
    parser.add_argument("--model_dir", type=str, required=True, help="Path to models directory")
    parser.add_argument("--data_dir", type=str, default="/home/local/data/sophie/node21_num_label_resample_all", help="Path to data directory")
    parser.add_argument("--log_dir", type=str, required=True, help="Tensorboard logging folder")

    return parser.parse_args()

def main():
    
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    writer = SummaryWriter(log_dir=args.log_dir)

    model = torchvision.models.resnet50(pretrained=True)

    # grab checkpoint for environmental configs
    env_conf = torch.load(os.path.join(args.model_dir, "checkpoint.pth"), map_location='cpu')['args']
    # data =============================================
    ext_names = ['cxr14', 'padchest', 'openi', 'jsrt']
    ext_names.remove(env_conf.train_set)

    mean = arch_seg_dict[env_conf.train_set][0]
    std = arch_seg_dict[env_conf.train_set][1]
    normalise = v2.Normalize(mean=mean, std=std)

    test_transform = get_cxr_eval_transforms(
        crop_size =  env_conf.crop_size, 
        normalise = normalise, 
        single = env_conf.model_type == "single")

    # sophie/node21_num_label_resample_all/cxr14/arch_seg/flat_std_1024/test/0_normal
    internal_test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(args.data_dir,env_conf.train_set,"arch_seg/flat_std_1024/test"), transform=v2.Compose(test_transform))
    ext1_test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(args.data_dir,ext_names[0],"arch_seg/flat_std_1024/test"), transform=v2.Compose(test_transform))
    ext2_test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(args.data_dir,ext_names[1],"arch_seg/flat_std_1024/test"), transform=v2.Compose(test_transform))
    ext3_test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(args.data_dir,ext_names[2],"arch_seg/flat_std_1024/test"), transform=v2.Compose(test_transform))

    internal_test_dataloader = torch.utils.data.DataLoader(internal_test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    ext1_test_dataloader = torch.utils.data.DataLoader(ext1_test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    ext2_test_dataloader = torch.utils.data.DataLoader(ext2_test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    ext3_test_dataloader = torch.utils.data.DataLoader(ext3_test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    for model_name in os.listdir(args.model_dir):

        # check is a .pth file
        if model_name[0][-4:] == ".pth":

            model_path = os.path.join(args.model_dir,model_name)

            model_object = torch.load(model_path, map_location='cpu')
            
            # copy pre-loaded model
            this_model = copy.deepcopy(model)

            # check if single channel model
            if model_object['args'].model_type == "single":

                this_model = convert_to_single_channel(this_model)

            # load model
            this_model.load_state_dict(model_object["model"])
            
            this_model.to(device)
            this_model.eval()

            # iterate over datasets
            for test_name, loader in zip(['test', 'ext1', 'ext2', 'ext3'], [testloader, ext1loader, ext2loader, ext3loader]):
                precision, recall, f1, auc = evaluate_gpu_metrics(this_model, loader, device)
                print(f'{test_name} - Epoch {epoch + 1}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, AUC={auc:.3f}')
                # Log these metrics to TensorBoard
                writer.add_scalar(f'Precision/{test_name}', precision, epoch + 1)
                writer.add_scalar(f'Recall/{test_name}', recall, epoch + 1)
                writer.add_scalar(f'F1/{test_name}', f1, epoch + 1)
                writer.add_scalar(f'AUC/{test_name}', auc, epoch + 1)

if __name__ == "__main__":
    main()