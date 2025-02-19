# python terminal_gridsearch.py --process 'crop' --train_set 'cxr14' --model_type 'grey' --tuning_strategy 'final_layer' --epochs 30 --patience 5 --learning_rates 0.001 0.01 0.01 --batch_sizes 16 --momentums 0.9 0.95 0.99 --data_root '/home/local/data/sophie/' --num_workers 4 --log_dr '/home/local/data/sophie/runs/128_runs/' --crop_size 128
import copy
import itertools
import torch
from torch.backends.cudnn import benchmark, deterministic
import torchvision
from torchvision.transforms import v2
import random
import numpy as np
# from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
# core metrics
# BinaryAUROC: summarise overall performance
# BinaryF1Score:
from torcheval.metrics import BinaryAUROC, BinaryF1Score, BinaryPrecision, BinaryRecall
# medical metrics - use torchmetrics package for specific calculations
# BinarySpecificityAtSensitivity: tune to 98% sensitivity to see performance of negatives when nearly all positives are detected
# BinaryFBetaScore: F-Score but with higher importance weighting to class (beta>1 = higher recall ~ identify all positives)
# from torchmetrics.classification import BinarySpecificityAtSensitivity, BinaryFBetaScore
# from torchmetrics.classification import BinaryPrecisionAtFixedRecall as BinaryPrecisionAtSensitivity # rename for clearer understanding
from torchvision.models import get_model
from torch.utils.tensorboard import SummaryWriter
import os
import datetime
import argparse



crop_dict = {
    # data      mean         std
    'cxr14': [[162.7414], [44.0700]],
    'openi': [[157.6150], [41.8371]],
    'jsrt': [[161.7889], [41.3950]],
    'padchest': [[160.3638], [44.8449]],
}

arch_seg_dict = {
    # data       mean        std
    'cxr14': [[128.2716], [76.7148]],
    'openi': [[127.7211], [69.7704]],
    'jsrt': [[139.9666], [72.4017]],
    'padchest': [[129.5006], [72.6308]],
}

lung_seg_dict = {
    # data       mean        std
    'cxr14': [[60.6809], [68.9660]],
    'openi': [[60.5483], [66.5276]],
    'jsrt': [[66.5978], [72.6493]],
    'padchest': [[60.5482], [66.5276]],
}
def get_cxr_train_transforms(crop_size,normalise,single=False):
    if single:
        grey = v2.Grayscale(num_output_channels=1)
    else:
        grey = v2.Grayscale(num_output_channels=3)
    cxr_transform_list = [
        v2.ToImage(),
        v2.RandomRotation(15),
        v2.RandomHorizontalFlip(),
        v2.RandomApply([
            # reduced saturation and contrast - prevent too much info loss + removed hue
            v2.ColorJitter(0.4, 0.2, 0.2,0)
        ], p=0.8),
        grey,
        # moved after transforms to preserve resolution, reduced scale to increase likelihood of indicator presence
        v2.RandomResizedCrop(size=crop_size, scale=(0.6, 1.),antialias=True),
        # required for normalisation
        v2.ToDtype(torch.float32, scale=False),
        normalise
    ]
    return cxr_transform_list

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
def set_seed(seed):
    """Set the seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def evaluate_model(model, dataloader, device):
    """Evaluate the model on the given dataloader and return performance metrics."""
    model.eval()
    true_labels = []
    pred_labels = []
    pred_probs = []

    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Get probabilities for class 1
            preds = torch.argmax(outputs, dim=1)

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())
            pred_probs.extend(probs.cpu().numpy())

    # Compute metrics for binary classification
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    auc = roc_auc_score(true_labels, pred_probs)

    return precision, recall, f1, auc

def evaluate_gpu_metrics(model, dataloader, device):
    model.eval()
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


def set_model_tuning(model, tuning_strategy):
    """Set the model's layers to be trainable or frozen based on the tuning strategy."""
    if tuning_strategy == "final_layer":
        # Freeze all layers except the final fully connected layer
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze the final layer
        for param in model.fc.parameters():
            param.requires_grad = True

    elif tuning_strategy == "half_network":
        # Get the total number of layers (parameters) in the model
        total_layers = len(list(model.parameters()))
        # Unfreeze the last half of the layers
        for i, param in enumerate(model.parameters()):
            if i >= total_layers // 2:
                param.requires_grad = True
            else:
                param.requires_grad = False

    elif tuning_strategy == "full_network":
        # Unfreeze all layers
        for param in model.parameters():
            param.requires_grad = True
    elif tuning_strategy == "first_layer_freeze":
        first_freeze = False
        # Unfreeze all layers
        for param in model.parameters():
            if first_freeze:
                param.requires_grad = True
            else:
                param.requires_grad = False
                first_freeze = True
        # # Refreeze first layer
        # model.conv1.parameters().requires_grad(False)
    else:
        raise ValueError("Invalid tuning strategy. Choose from 'final_layer', 'half_network', 'first_layer_freeze' or 'full_network'.")

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
        if isinstance(layer, torch.nn.Conv2d):
            conv1 = layer
            conv1_name = name
            break

    if conv1 is None:
        raise ValueError("The model does not have a Conv2D layer.")

    # Create a new convolutional layer with the same parameters except for the input channels
    new_conv1 = torch.nn.Conv2d(
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

# def run_model_training(crop_size, process, train_set, model, model_name, bsz, lr, momentum, patience, tuning_strategy, log_dr,data_root="/content/", num_epochs=10,num_workers=8, single=False, seed=None,pos_class_weight=1.0,target_mom=None,target_mom_rate=None):
def run_model_training(model, bsz, lr, momentum, seed, pos_class_weight, target_mom, target_mom_rate, args):
    """
    Train the model and log results to TensorBoard, organizing logs by tuning strategy, model, and hyperparameters.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    class_weighting = torch.Tensor([1.0,pos_class_weight]).to(device)

    single= True if "single" in args.model_type else False

    if target_mom is not None:
        # Create a log directory based on the tuning strategy, model name, and hyperparameters
        log_dir = os.path.join(
            # "/content/drive/MyDrive/alignment/runs",
            args.log_dir,
            f"linmom_{target_mom}_{target_mom_rate}",
            args.process,
            args.tuning_strategy,  # Directory for the tuning strategy
            args.model_type,  # Subdirectory for the model type
            f"lr_{lr}_bsz_{bsz}_mom_{momentum}_seed_{seed}_posWeight_{pos_class_weight}"  # Subdirectory for hyperparameter configuration
        )
    else:
        # Create a log directory based on the tuning strategy, model name, and hyperparameters
        log_dir = os.path.join(
            # "/content/drive/MyDrive/alignment/runs",
            args.log_dir,
            args.process,
            args.tuning_strategy,  # Directory for the tuning strategy
            args.model_type,  # Subdirectory for the model type
            f"lr_{lr}_bsz_{bsz}_mom_{momentum}_seed_{seed}_posWeight_{pos_class_weight}"  # Subdirectory for hyperparameter configuration
        )

    # Ensure the directory structure is created properly
    os.makedirs(log_dir, exist_ok=True)

    # Initialize TensorBoard writer with the dynamically generated log directory
    writer = SummaryWriter(log_dir=log_dir)

    ext_names = ['cxr14', 'padchest', 'openi', 'jsrt']
    if args.train_set not in ext_names:
        raise ValueError("Invalid train_set value. Must be one of 'cxr14', 'padchest', 'openi', or 'jsrt'.")
    else:
        ext_names.remove(args.train_set)

    std_dir = "std_1024"
    if args.process == "crop":
        mean = crop_dict[args.train_set][0]
        std = crop_dict[args.train_set][1]
    elif args.process == "arch_seg":
        mean = arch_seg_dict[args.train_set][0]
        std = arch_seg_dict[args.train_set][1]
        std_dir = "flat_std_1024"
    elif args.process == "lung_seg":
        mean = lung_seg_dict[args.train_set][0]
        std = lung_seg_dict[args.train_set][1]
    else:
        raise ValueError("Invalid process value. Must be 'crop', 'arch_seg', or 'lung_seg'.")


    normalise = v2.Normalize(mean=mean, std=std)

    # Set the tuning strategy (which layers to freeze/unfreeze)
    set_model_tuning(model, args.tuning_strategy)

    # Data transformations
    train_transform = get_cxr_train_transforms(args.crop_size, normalise, single)
    test_transform = get_cxr_eval_transforms(args.crop_size, normalise, single)

    # Load datasets
    train_path = os.path.join(args.data_root, args.train_set, args.process, std_dir, "train")
    test_path = os.path.join(args.data_root, args.train_set, args.process, std_dir, "test")
    ext1_path = os.path.join(args.data_root, ext_names[0], args.process, std_dir, "test")
    ext2_path = os.path.join(args.data_root, ext_names[1], args.process, std_dir, "test")
    ext3_path = os.path.join(args.data_root, ext_names[2], args.process, std_dir, "test")

    def remap_labels(label):
        # mapping_dict = {'normal': 0, 'nodule': 1}
        mapping_dict = {trainset.class_to_idx['nodule']: 1, trainset.class_to_idx['normal']: 0}
        return mapping_dict[label]
    trainset = torchvision.datasets.ImageFolder(root=train_path, transform= v2.Compose(train_transform), target_transform=remap_labels)
    testset = torchvision.datasets.ImageFolder(root=test_path, transform=v2.Compose(test_transform), target_transform=remap_labels)
    ext1set = torchvision.datasets.ImageFolder(root=ext1_path, transform=v2.Compose(test_transform), target_transform=remap_labels)
    ext2set = torchvision.datasets.ImageFolder(root=ext2_path, transform=v2.Compose(test_transform), target_transform=remap_labels)
    ext3set = torchvision.datasets.ImageFolder(root=ext3_path, transform=v2.Compose(test_transform), target_transform=remap_labels)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bsz, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=bsz, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    ext1loader = torch.utils.data.DataLoader(ext1set, batch_size=bsz, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    ext2loader = torch.utils.data.DataLoader(ext2set, batch_size=bsz, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    ext3loader = torch.utils.data.DataLoader(ext3set, batch_size=bsz, shuffle=False, num_workers=args.num_workers, pin_memory=True)



    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss(weight=class_weighting)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    model.to(device)

    best_loss = float('inf')
    epochs_no_improve = 0
    # final_metrics = {}
    # Track metrics
    metrics_dict = {
        "epoch": [],
        "train_loss": [],
        "test_auc": [],
        "test_precision": [],
        "test_recall": [],
        "test_f1": [],
        "ext1_auc": [],
        "ext1_precision": [],
        "ext1_recall": [],
        "ext1_f1": [],
        "ext2_auc": [],
        "ext2_precision": [],
        "ext2_recall": [],
        "ext2_f1": [],
        "ext3_auc": [],
        "ext3_precision": [],
        "ext3_recall": [],
        "ext3_f1": [],

    }

    print(f"Training model: {args.model_type} for {args.epochs} epochs with seed: {torch.initial_seed()}")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        # enable linear scaling momentum
        if target_mom is not None:
            # check if momentum needs increasing
            if momentum < target_mom:
                # increase
                momentum += target_mom_rate
                # check if addition has exceeded target
                if momentum > target_mom:
                    momentum = target_mom
                # update momentum
                for param_group in optimizer.param_groups:
                    param_group['momentum'] = momentum

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(trainloader)
        print(f'Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss:.3f}')
        writer.add_scalar('Loss/train', avg_loss, epoch + 1)
        metrics_dict["epoch"].append(epoch + 1)
        metrics_dict["train_loss"].append(avg_loss)

        #save model 
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": args,
        }
        out_model_path = os.path.join(
            args.output_dir,
            args.process,
            args.tuning_strategy,  # Directory for the tuning strategy
            args.model_type,  # Subdirectory for the model type
        )
    
        out_model_name = f"model_lr_{lr}_bsz_{bsz}_mom_{momentum}_seed_{seed}_pos-weight_{pos_class_weight}.pth"
        if target_mom is not None:
            out_model_path = os.path.join(out_model_path,f"linmom_{target_mom}_{target_mom_rate}")
    
        if not os.path.exists(out_model_path):
            os.makedirs(out_model_path)

        torch.save(checkpoint, os.path.join(out_model_path, f"model_{epoch}.pth"))
        torch.save(checkpoint, os.path.join(out_model_path, "checkpoint.pth"))
        # Evaluate on all test sets
        for test_name, loader in zip(['test', 'ext1', 'ext2', 'ext3'], [testloader, ext1loader, ext2loader, ext3loader]):
            # precision, recall, f1, auc = evaluate_model(model, loader, device)
            precision, recall, f1, auc = evaluate_gpu_metrics(model, loader, device)

            print(f'{test_name} - Epoch {epoch + 1}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, AUC={auc:.3f}')

            # Log these metrics to TensorBoard
            writer.add_scalar(f'Precision/{test_name}', precision, epoch + 1)
            writer.add_scalar(f'Recall/{test_name}', recall, epoch + 1)
            writer.add_scalar(f'F1/{test_name}', f1, epoch + 1)
            writer.add_scalar(f'AUC/{test_name}', auc, epoch + 1)

            # Save the metrics for post-training analysis

            metrics_dict[f'{test_name}_precision'].append(precision)
            metrics_dict[f'{test_name}_recall'].append(recall)
            metrics_dict[f'{test_name}_f1'].append(f1)
            metrics_dict[f'{test_name}_auc'].append(auc)

        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print(f"Early stopping after {epochs_no_improve} epochs with no improvement.")
            break

    writer.close()

    # final_model = {
    #     "model": model.state_dict(),
    #     "optimizer": optimizer.state_dict(),
    #     "epoch": epoch,
    # }
    # out_model_path = os.path.join(
    #     args.output_dir,
    #     args.process,
    #     args.tuning_strategy,  # Directory for the tuning strategy
    #     args.model_type,  # Subdirectory for the model type
    # )

    # out_model_name = f"model_lr_{lr}_bsz_{bsz}_mom_{momentum}_seed_{seed}_pos-weight_{pos_class_weight}.pth"
    # if target_mom is not None:
    #     out_model_path = os.path.join(out_model_path,f"linmom_{target_mom}_{target_mom_rate}")

    # if not os.path.exists(out_model_path):
    #     os.makedirs(out_model_path)

    # torch.save(final_model, os.path.join(out_model_path,out_model_name))
    return metrics_dict

# def grid_search(crop_size, process, train_set, model, model_name, patience, param_grid, tuning_strategy, num_epochs=10, data_root="/content/",num_workers=8,log_dr="runs",single=False,tgt_mom_epoch=None):
def grid_search(model, param_grid, args):
    """Perform a grid search to identify the best hyperparameter configuration."""
    best_params = None
    best_score = float('-inf')

    # handle unset/unwanted flags
    for key in list(param_grid.keys()):
        if param_grid.get(key) is None:
            param_grid.pop(key)


    param_combinations = list(itertools.product(*param_grid.values()))
    param_names = list(param_grid.keys())

    print(f"Total parameter combinations: {len(param_combinations)}")

    for idx, param_comb in enumerate(param_combinations):
        params = dict(zip(param_names, param_comb))
        bsz, lr, momentum, seed, pos_class_weight = params['bsz'], params['lr'], params['momentum'], params['seed'], params['pos_class_weights']

        if args.target_momentums_epoch is not None:
            tgt_mom = params['target_momentum']
            tgt_mom_rate = (args.target_momentums_epoch-momentum)/args.target_momentums_epoch
        else:
            tgt_mom_rate = None
            tgt_mom = None

        print(f"Grid search iteration {idx + 1}/{len(param_combinations)} with params: {params} on model: {args.model_type}")
        set_seed(seed)

        model_copy = copy.deepcopy(model)

        # metrics = run_model_training(crop_size, process, train_set, model_copy, model_name, bsz, lr, momentum, patience, tuning_strategy, log_dr, data_root, num_epochs, num_workers, single,seed=seed,pos_class_weight=pos_class_weight,target_mom=tgt_mom,target_mom_rate=tgt_mom_rate)
        metrics = run_model_training(model_copy, bsz, lr, momentum, seed, pos_class_weight, tgt_mom_rate, tgt_mom, args)

        avg_auc = (metrics['test_auc'][-1] + metrics['ext1_auc'][-1] + metrics['ext2_auc'][-1] + metrics['ext3_auc'][-1]) / 4


        if avg_auc > best_score:
            best_score = avg_auc
            best_params = params

        print(f"Avg AUC: {avg_auc:.3f} (Best so far: {best_score:.3f})")
        torch.cuda.empty_cache()

    print(f"Best hyperparameters: {best_params}, AUC: {best_score:.3f}")
    return best_params


def parse_args():
    parser = argparse.ArgumentParser(description="Run grid search for model training")

    # Add arguments for the grid search
    parser.add_argument("--crop_size", type=int, default=224, help="Crop size for image preprocessing")
    parser.add_argument("--process", type=str, required=True, choices=["crop", "arch_seg", "lung_seg"], help="Process type")
    parser.add_argument("--train_set", type=str, required=True, help="Training dataset name (e.g., 'cxr14')")
    # parser.add_argument("--model_type", type=str, required=True, choices=["base", "grey","grey89", "final_base", "final_grey", "final_single","final_single_last","final_base99", "final_grey99",], help="Model type (base or grey)")
    parser.add_argument("--model_type", type=str, required=True, \
                        choices=['grey', 'base', 'single', 'grey65', 'base65', 'single65'], \
                        help="Model type ['grey', 'base', 'single', 'grey65', 'base65', 'single65']")
    parser.add_argument("--tuning_strategy", type=str, required=True, choices=["final_layer", "half_network", "first_layer_freeze","full_network"], help="Tuning strategy")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")

    # Add arguments for hyperparameter grid (can be passed as a list)
    parser.add_argument("--learning_rates", nargs='+', type=float, default=[0.001, 0.01, 0.1], help="List of learning rates")
    parser.add_argument("--batch_sizes", nargs='+', type=int, default=[16, 32, 64], help="List of batch sizes")
    parser.add_argument("--momentums", nargs='+', type=float, default=[0.9, 0.95, 0.99], help="List of momentums")
    parser.add_argument("--target_momentums", nargs='+', type=float, help="Target momentums to reach (assumes momentums has been set as initial momentum).")
    parser.add_argument("--target_momentums_epoch", type=float, help="Epoch target momentum should be reached.")



    # Add args for experiment config
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for dataloader")
    parser.add_argument("--data_root", type=str, required=True, help="Root data folder (contains 'cxr14'...")
    parser.add_argument("--log_dir", type=str, default="runs", help="Tensorboard logging folder")
    parser.add_argument("--output_dir", type=str, default="models", help="Model save folder")

    # Optional seed
    parser.add_argument("--seed", nargs='+', type=int, default=42, help="Random seed - supports list")

    # Class weights
    parser.add_argument("--pos_class_weights", nargs='+', type=float, default=[1.0], help="List of weights to eval for positive class. Negative weight left as 1.0")

    return parser.parse_args()


def initialize_model(model_type):
    num_classes = 2
    """Initialize and return the model based on the type specified."""
    if model_type == "base":
        # model = torchvision.models.resnet50(pretrained=True)
        weights = torch.load("/home/local/data/sophie/imagenet/output/FullGPU/base/256/model_89.pth", map_location='cpu', weights_only=False)
        model = get_model("resnet50",weights=None,num_classes=1000)
        model.load_state_dict(weights["model"])
    elif model_type == "grey":
        # model = torchvision.models.resnet50(pretrained=True)
        # weights = torch.load("/home/local/data/sophie/model_75.pth", map_location='cpu')
        # new_state_dict = {}
        # for k, v in weights['model'].items():
        #     k = k.replace("module.", "")
        #     new_state_dict[k] = v
        # model.load_state_dict(new_state_dict)
        weights = torch.load("/home/local/data/sophie/imagenet/output/FullGPU/grey/256/model_89.pth", map_location='cpu', weights_only=False)
        model = get_model("resnet50",weights=None,num_classes=1000)
        model.load_state_dict(weights["model"])
    elif model_type == "single":
        weights = torch.load("/home/local/data/sophie/imagenet/output/FullGPU/single/256/model_89.pth", map_location='cpu', weights_only=False)
        model = get_model("resnet50",weights=None,num_classes=1000)
        model = convert_to_single_channel(model)
        model.load_state_dict(weights["model"])
    elif model_type == "base65":
        weights = torch.load("/home/local/data/sophie/imagenet/output/FullGPU/base/256/model_64.pth", map_location='cpu', weights_only=False)
        model = get_model("resnet50",weights=None,num_classes=1000)
        model.load_state_dict(weights["model"])
    elif model_type == "single65":
        weights = torch.load("/home/local/data/sophie/imagenet/output/FullGPU/single/256/model_64.pth", map_location='cpu', weights_only=False)
        model = get_model("resnet50",weights=None,num_classes=1000)
        model = convert_to_single_channel(model)
        model.load_state_dict(weights["model"])
    elif model_type == "grey65":
        weights = torch.load("/home/local/data/sophie/imagenet/output/FullGPU/grey/256/model_64.pth", map_location='cpu', weights_only=False)
        model = get_model("resnet50",weights=None,num_classes=1000)
        model.load_state_dict(weights["model"])
    else:
        raise ValueError("Invalid Model Choice! Choose from ['grey', 'base', 'single', 'grey65', 'base65', 'single65']")
    # elif model_type == "grey89":
    #     model = torchvision.models.resnet50(pretrained=True)
    #     weights = torch.load("/home/local/data/sophie/model_89.pth", map_location='cpu')
    #     new_state_dict = {}
    #     for k, v in weights['model'].items():
    #         k = k.replace("module.", "")
    #         new_state_dict[k] = v
    #     model.load_state_dict(new_state_dict)
    # elif model_type == "final_base":
    #     weights = torch.load("/home/local/data/sophie/imagenet/output/FullGPU/base/continued/model_150.pth", map_location='cpu', weights_only=False)
    #     model = get_model("resnet50",weights=None,num_classes=1000)
    #     model.load_state_dict(weights["model"])
    # elif model_type == "final_grey":
    #     weights = torch.load("/home/local/data/sophie/imagenet/output/FullGPU/grey/continued/model_151.pth", map_location='cpu', weights_only=False)
    #     model = get_model("resnet50",weights=None,num_classes=1000)
    #     model.load_state_dict(weights["model"])
    # elif model_type == "final_base99":
    #     weights = torch.load("/home/local/data/sophie/imagenet/output/FullGPU/base/continued/model_99.pth", map_location='cpu', weights_only=False)
    #     model = get_model("resnet50",weights=None,num_classes=1000)
    #     model.load_state_dict(weights["model"])
    # elif model_type == "final_grey99":
    #     weights = torch.load("/home/local/data/sophie/imagenet/output/FullGPU/grey/continued/model_99.pth", map_location='cpu', weights_only=False)
    #     model = get_model("resnet50",weights=None,num_classes=1000)
    #     model.load_state_dict(weights["model"])
    # elif model_type == "final_single":
    #     weights = torch.load("/home/local/data/sophie/imagenet/output/FullGPU/single/model_99.pth", map_location='cpu', weights_only=False)
    #     model = get_model("resnet50",weights=None,num_classes=1000)
    #     model = convert_to_single_channel(model)
    #     model.load_state_dict(weights["model"])
    # elif model_type == "final_single_last":
    #     weights = torch.load("/home/local/data/sophie/imagenet/output/FullGPU/single/model_120.pth", map_location='cpu', weights_only=False)
    #     model = get_model("resnet50",weights=None,num_classes=1000)
    #     model = convert_to_single_channel(model)
    #     model.load_state_dict(weights["model"])
    

    # ensure results are consistent by disabling benchmarking
    benchmark = False
    deterministic = True


    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    return model

def main():
    args = parse_args()
    #stop crazy CPU core use
    torch.set_num_threads(args.num_workers)
    # Initialize the model based on the type ('grey', 'base', 'single', 'grey65', 'base65', 'single65')
    model = initialize_model(args.model_type)

    if args.target_momentums is not None:
        for tg_mom in args.target_momentums:
            for mom in args.momentums:
                if tg_mom <= mom:
                    raise ValueError("Target momentum ({}) must be more than momentum{}!".format(tg_mom,mom))
        if args.target_momentums_epoch is None:
            raise ValueError("Target momentum epochs (--target_momentums_epoch) must be set with target momentum!")
        else:
            if args.target_momentums_epoch > args.epochs:
                raise ValueError("Target momentum epoch {} must be less than or equal to the total number of training epochs{}!".format(args.target_momentums_epoch, args.epochs))
    # Hyperparameter grid for grid search
    param_grid = {
        'bsz': args.batch_sizes,
        'lr': args.learning_rates,
        'momentum': args.momentums,
        'target_momentum': args.target_momentums,
        'seed': args.seed,
        'pos_class_weights': args.pos_class_weights
    }

    # Run the grid search
    # best_params = grid_search(
    #     crop_size=args.crop_size,
    #     process=args.process,
    #     train_set=args.train_set,
    #     model=model,
    #     model_name=args.model_type,
    #     patience=args.patience,
    #     param_grid=param_grid,
    #     tuning_strategy=args.tuning_strategy,
    #     num_epochs=args.epochs,
    #     data_root=args.data_root,
    #     num_workers=args.num_workers,
    #     log_dr=args.log_dr,
    #     single=True if "single" in args.model_type else False,
    #     tgt_mom_epoch = args.target_momentums_epoch,
    # )
    best_params = grid_search(
        model=model,
        param_grid=param_grid,
        args=args
    )

    print(f"Best parameters found: {best_params}")

if __name__ == "__main__":
    main()
