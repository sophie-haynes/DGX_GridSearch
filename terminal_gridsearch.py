# python terminal_gridsearch.py --process 'crop' --train_set 'cxr14' --model_type 'grey' --tuning_strategy 'final_layer' --epochs 30 --patience 5 --learning_rates 0.001 0.01 0.01 --batch_sizes 16 --momentums 0.9 0.95 0.99 --data_root '/home/local/data/sophie/' --num_workers 4 --log_dr '/home/local/data/sophie/runs/128_runs/' --crop_size 128
import copy
import itertools
import torch
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
def get_cxr_train_transforms(crop_size,normalise):
    cxr_transform_list = [
        v2.ToImage(),
        v2.RandomRotation(15),
        v2.RandomHorizontalFlip(),
        v2.RandomApply([
            # reduced saturation and contrast - prevent too much info loss + removed hue
            v2.ColorJitter(0.4, 0.2, 0.2,0)
        ], p=0.8),
        # moved after transforms to preserve resolution, reduced scale to increase likelihood of indicator presence
        v2.RandomResizedCrop(size=crop_size, scale=(0.6, 1.),antialias=True),
        # required for normalisation
        v2.ToDtype(torch.float32, scale=False),
        normalise
    ]
    return cxr_transform_list

def get_cxr_eval_transforms(crop_size,normalise):
    cxr_transform_list = [
        v2.ToImage(),
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

    # prec98 = BinaryPrecisionAtSensitivity(min_recall=0.98)
    # spec98 = BinarySpecificityAtSensitivity(min_sensitivity=0.98)
    # fbeta = BinaryFBetaScore(beta=2.0)

    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            auroc.update(output,labels)
            f1.update(output,labels)
            prec.update(output,labels)
            rec.update(output,labels)


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


def run_model_training(crop_size, process, train_set, model, model_name, bsz, lr, momentum, patience, tuning_strategy, log_dr,data_root="/content/", num_epochs=10,num_workers=8):
    """
    Train the model and log results to TensorBoard, organizing logs by tuning strategy, model, and hyperparameters.
    """

    # Create a log directory based on the tuning strategy, model name, and hyperparameters
    log_dir = os.path.join(
        # "/content/drive/MyDrive/alignment/runs",
        log_dr,
        process,
        tuning_strategy,  # Directory for the tuning strategy
        model_name,  # Subdirectory for the model type
        f"lr_{lr}_bsz_{bsz}_mom_{momentum}"  # Subdirectory for hyperparameter configuration
    )

    # Ensure the directory structure is created properly
    os.makedirs(log_dir, exist_ok=True)

    # Initialize TensorBoard writer with the dynamically generated log directory
    writer = SummaryWriter(log_dir=log_dir)

    ext_names = ['cxr14', 'padchest', 'openi', 'jsrt']
    if train_set not in ext_names:
        raise ValueError("Invalid train_set value. Must be one of 'cxr14', 'padchest', 'openi', or 'jsrt'.")
    else:
        ext_names.remove(train_set)

    std_dir = "std_1024"
    if process == "crop":
        mean = crop_dict[train_set][0]
        std = crop_dict[train_set][1]
    elif process == "arch_seg":
        mean = arch_seg_dict[train_set][0]
        std = arch_seg_dict[train_set][1]
        std_dir = "flat_std_1024"
    elif process == "lung_seg":
        mean = lung_seg_dict[train_set][0]
        std = lung_seg_dict[train_set][1]
    else:
        raise ValueError("Invalid process value. Must be 'crop', 'arch_seg', or 'lung_seg'.")


    normalise = v2.Normalize(mean=mean, std=std)

    # Set the tuning strategy (which layers to freeze/unfreeze)
    set_model_tuning(model, tuning_strategy)

    # Data transformations
    train_transform = get_cxr_train_transforms(crop_size, normalise)
    test_transform = get_cxr_eval_transforms(crop_size, normalise)

    # Load datasets
    train_path = os.path.join(data_root, train_set, process, std_dir, "train")
    test_path = os.path.join(data_root, train_set, process, std_dir, "test")
    ext1_path = os.path.join(data_root, ext_names[0], process, std_dir, "test")
    ext2_path = os.path.join(data_root, ext_names[1], process, std_dir, "test")
    ext3_path = os.path.join(data_root, ext_names[2], process, std_dir, "test")

    trainset = torchvision.datasets.ImageFolder(root=train_path, transform= v2.Compose(train_transform))
    testset = torchvision.datasets.ImageFolder(root=test_path, transform=v2.Compose(test_transform))
    ext1set = torchvision.datasets.ImageFolder(root=ext1_path, transform=v2.Compose(test_transform))
    ext2set = torchvision.datasets.ImageFolder(root=ext2_path, transform=v2.Compose(test_transform))
    ext3set = torchvision.datasets.ImageFolder(root=ext3_path, transform=v2.Compose(test_transform))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bsz, shuffle=True, num_workers=num_workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=bsz, shuffle=False, num_workers=num_workers, pin_memory=True)
    ext1loader = torch.utils.data.DataLoader(ext1set, batch_size=bsz, shuffle=False, num_workers=num_workers, pin_memory=True)
    ext2loader = torch.utils.data.DataLoader(ext2set, batch_size=bsz, shuffle=False, num_workers=num_workers, pin_memory=True)
    ext3loader = torch.utils.data.DataLoader(ext3set, batch_size=bsz, shuffle=False, num_workers=num_workers, pin_memory=True)



    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    print(f"Training model: {model_name} for {num_epochs} epochs with seed: {torch.initial_seed()}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(trainloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.3f}')
        writer.add_scalar('Loss/train', avg_loss, epoch + 1)
        metrics_dict["epoch"].append(epoch + 1)
        metrics_dict["train_loss"].append(avg_loss)
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

        if epochs_no_improve >= patience:
            print(f"Early stopping after {epochs_no_improve} epochs with no improvement.")
            break

    writer.close()
    return metrics_dict

def grid_search(crop_size, process, train_set, model, model_name, patience, param_grid, seed, tuning_strategy, num_epochs=10, data_root="/content/",num_workers=8,log_dr="runs"):
    """Perform a grid search to identify the best hyperparameter configuration."""
    best_params = None
    best_score = float('-inf')

    param_combinations = list(itertools.product(*param_grid.values()))
    param_names = list(param_grid.keys())

    print(f"Total parameter combinations: {len(param_combinations)}")

    for idx, param_comb in enumerate(param_combinations):
        params = dict(zip(param_names, param_comb))
        bsz, lr, momentum = params['bsz'], params['lr'], params['momentum']

        print(f"Grid search iteration {idx + 1}/{len(param_combinations)} with params: {params} on model: {model_name}")
        set_seed(seed)

        model_copy = copy.deepcopy(model)

        metrics = run_model_training(crop_size, process, train_set, model_copy, model_name, bsz, lr, momentum, patience, tuning_strategy, log_dr, data_root, num_epochs,num_workers)

        avg_auc = (metrics['test_auc'][-1] + metrics['ext1_auc'][-1] + metrics['ext2_auc'][-1] + metrics['ext3_auc'][-1]) / 4


        if avg_auc > best_score:
            best_score = avg_auc
            best_params = params

        print(f"Avg AUC: {avg_auc:.3f} (Best so far: {best_score:.3f})")
        torch.cuda.empty_cache()

    print(f"Best hyperparameters: {best_params}, AUC: {best_score:.3f}")
    return best_params

def validate_best_params(crop_size, process, train_set, model, model_name, best_params, patience, seeds, tuning_strategy, log_dr, num_epochs=10, data_root="/content/"):
    """Validate the best hyperparameters across multiple seeds."""
    bsz = best_params['bsz']
    lr = best_params['lr']
    momentum = best_params['momentum']

    results = {'test_auc': [], 'ext1_auc': [], 'ext2_auc': [], 'ext3_auc': []}

    for i, seed in enumerate(seeds):
        print(f"Validation with seed {seed} (Run {i+1}/{len(seeds)}) with best params: {best_params}")
        set_seed(seed)

        metrics = run_model_training(crop_size, process, train_set, model, model_name, bsz, lr, momentum, patience, tuning_strategy, log_dr, data_root, num_epochs)
        results['test_auc'].append(metrics['test_auc'])
        results['ext1_auc'].append(metrics['ext1_auc'])
        results['ext2_auc'].append(metrics['ext2_auc'])
        results['ext3_auc'].append(metrics['ext3_auc'])

    avg_results = {key: np.mean(values) for key, values in results.items()}
    print(f"Average performance across seeds: {avg_results}")
    return avg_results


def parse_args():
    parser = argparse.ArgumentParser(description="Run grid search for model training")

    # Add arguments for the grid search
    parser.add_argument("--crop_size", type=int, default=224, help="Crop size for image preprocessing")
    parser.add_argument("--process", type=str, required=True, choices=["crop", "arch_seg", "lung_seg"], help="Process type")
    parser.add_argument("--train_set", type=str, required=True, help="Training dataset name (e.g., 'cxr14')")
    parser.add_argument("--model_type", type=str, required=True, choices=["base", "grey","grey89"], help="Model type (base or grey)")
    parser.add_argument("--tuning_strategy", type=str, required=True, choices=["final_layer", "half_network", "first_layer_freeze","full_network"], help="Tuning strategy")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")

    # Add arguments for hyperparameter grid (can be passed as a list)
    parser.add_argument("--learning_rates", nargs='+', type=float, default=[0.001, 0.01, 0.1], help="List of learning rates")
    parser.add_argument("--batch_sizes", nargs='+', type=int, default=[16, 32, 64], help="List of batch sizes")
    parser.add_argument("--momentums", nargs='+', type=float, default=[0.9, 0.95, 0.99], help="List of momentums")


    # Add args for experiment config
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for dataloader")
    parser.add_argument("--data_root", type=str, required=True, help="Root data folder (contains 'cxr14'...")
    parser.add_argument("--log_dr", type=str, help="Tensorboard logging folder")

    # Optional seed
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def initialize_model(model_type):

    """Initialize and return the model based on the type specified."""
    if model_type == "base":
        model = torchvision.models.resnet50(pretrained=True)
    elif model_type == "grey":
        model = torchvision.models.resnet50(pretrained=True)
        weights = torch.load("/home/local/data/sophie/model_75.pth", map_location='cpu')
        new_state_dict = {}
        for k, v in weights['model'].items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    elif model_type == "grey89":
        model = torchvision.models.resnet50(pretrained=True)
        weights = torch.load("/home/local/data/sophie/model_89.pth", map_location='cpu')
        new_state_dict = {}
        for k, v in weights['model'].items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    num_classes = 2

    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    return model

def main():
    args = parse_args()
    #stop crazy CPU core use
    torch.set_num_threads(args.num_workers)
    # Initialize the model based on the type ('base' or 'grey')
    model = initialize_model(args.model_type)

    # Hyperparameter grid for grid search
    param_grid = {
        'bsz': args.batch_sizes,
        'lr': args.learning_rates,
        'momentum': args.momentums
    }

    # Run the grid search
    best_params = grid_search(
        crop_size=args.crop_size,
        process=args.process,
        train_set=args.train_set,
        model=model,
        model_name=args.model_type,
        patience=args.patience,
        param_grid=param_grid,
        seed=args.seed,
        tuning_strategy=args.tuning_strategy,
        num_epochs=args.epochs,
        data_root=args.data_root,
        num_workers=args.num_workers,
        log_dr=args.log_dr
    )

    print(f"Best parameters found: {best_params}")

if __name__ == "__main__":
    main()
