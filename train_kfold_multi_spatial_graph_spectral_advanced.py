"""
Train the spatial-spectral graph transformer (MVTSpatialSpectralModel) with
stratified k-fold cross-validation on the chunked ds004504 EEG data.

Example (full run, GPU):
    python train_kfold_multi_spatial_graph_spectral_advanced.py --data-dir model-data

Example (smoke test, CPU, a few minutes):
    python train_kfold_multi_spatial_graph_spectral_advanced.py --device cpu \
        --epochs 1 --folds 1 --max-samples-per-class 16 --batch-size 4 --num-workers 0
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import json
import random
import numpy as np
import warnings
import os
from datetime import datetime
import time
from contextlib import nullcontext
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support
import torch.nn.functional as F

# Import custom models and dataset
from eeg_multi_spatial_graph_spectral_advanced import create_model, MVTSpatialSpectralModel
from eeg_dataset_multispatialgraph_spectral_advanced import SpatialSpectralEEGDataset

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def progress_bar(current, total, width=50, prefix='', suffix=''):
    percent = f"{100 * (current / float(total)):.1f}%"
    filled_length = int(width * current // total)
    bar = '=' * filled_length + '-' * (width - filled_length)
    print(f'\r{prefix} [{bar}] {percent} {suffix}', end='')
    if current == total:
        print()

def get_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if not checkpoints:
        return None
    checkpoint_info = [(int(f.split('fold')[1].split('_')[0]), int(f.split('epoch')[1].split('.')[0]), f) 
                       for f in checkpoints if 'fold' in f and 'epoch' in f]
    return max(checkpoint_info, key=lambda x: (x[0], x[1]), default=None)

def compute_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    return precision, recall, f1

def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', default=os.environ.get('EEG_DATA_DIR', 'model-data'),
                        help='Folder produced by data_prep.py (contains labels.json, train/, test/). '
                             'Default: $EEG_DATA_DIR or ./model-data')
    parser.add_argument('--output-root', default='.', help='Where the spatial_spectral_<timestamp>/ run folder is created')
    parser.add_argument('--dim', type=int, default=128,
                        help='Model width. 128 = best cross-subject run (1.67M params); 264 was tried later (7.1M params)')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--folds', type=int, default=5, help='Number of CV folds to actually train (out of 5)')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--max-grad-norm', type=float, default=0.3)
    parser.add_argument('--time-length', type=int, default=2000, help='Samples per chunk after pad/truncate')
    parser.add_argument('--patience', type=int, default=30, help='Early-stopping patience (epochs)')
    parser.add_argument('--batch-size', type=int, default=None, help='Default: picked from free GPU memory (4 on CPU)')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--max-samples-per-class', type=int, default=None,
                        help='Cap the balanced training set per class (for quick smoke tests)')
    parser.add_argument('--device', default=None, help="'cuda' or 'cpu' (default: cuda if available)")
    parser.add_argument('--wandb', action='store_true', help='Log to Weights & Biases (requires `wandb login`)')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Model and training parameters
    num_channels = 19
    num_classes = 3
    dim = args.dim
    scales = [3, 4, 5]
    epochs = args.epochs
    learning_rate = args.lr
    weight_decay = args.weight_decay
    max_grad_norm = args.max_grad_norm
    time_length = args.time_length
    n_splits = 5

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    use_amp = device.type == 'cuda'

    # Dynamic batch size based on GPU memory
    if args.batch_size:
        batch_size = args.batch_size
    elif device.type == 'cuda':
        free_mem, total_mem = torch.cuda.mem_get_info()
        batch_size = 32 if free_mem > 8e9 else 16 if free_mem > 4e9 else 8  # Adjust based on 8GB, 4GB thresholds
    else:
        batch_size = 4
    accumulation_steps = max(1, 32 // batch_size)  # Ensure effective batch size ~32

    # Weights & Biases is optional
    wandb = None
    if args.wandb:
        import wandb

    def wandb_log(payload):
        if wandb is not None:
            wandb.log(payload)

    run_name = f"spatial_spectral_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if wandb is not None:
        wandb.init(project="eeg_spatial_spectral", name=run_name, config={
        "model_type": "MVTSpatialSpectral_Enhanced",
        "dim": dim,
        "scales": scales,
        "num_channels": num_channels,
        "num_classes": num_classes,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "accumulation_steps": accumulation_steps,
        "max_grad_norm": max_grad_norm,
        "optimizer": "AdamW",
        "time_length": time_length
    })

    # Directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(args.output_root, f'spatial_spectral_{timestamp}')
    for dir_path in [f'{base_dir}/models', f'{base_dir}/logs', f'{base_dir}/checkpoints']:
        os.makedirs(dir_path, exist_ok=True)

    log_filename = f'{base_dir}/logs/training_log.txt'
    def log_message(message):
        print(message)
        with open(log_filename, 'a', encoding='utf-8') as f:
            f.write(message + '\n')

    # Initialize model
    model = create_model(num_channels=num_channels, num_classes=num_classes, dim=dim, scales=scales)
    log_message(f"Using enhanced MVTSpatialSpectral model with scales {scales}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_message(f"Total trainable parameters: {total_params:,}")

    # Device setup
    model.to(device)
    log_message(f"Using device: {device}")
    if device.type == 'cuda':
        log_message(f"CUDA device: {torch.cuda.get_device_name(0)}")

    if wandb is not None:
        wandb.watch(model, log="all", log_freq=10)

    # Load data
    data_dir = args.data_dir
    with open(os.path.join(data_dir, 'labels.json'), 'r') as file:
        data_info = json.load(file)
    train_data = [d for d in data_info if d['type'] == 'train']

    # Balance dataset
    train_data_A = [d for d in train_data if d['label'] == 'A']
    train_data_C = [d for d in train_data if d['label'] == 'C']
    train_data_F = [d for d in train_data if d['label'] == 'F']
    min_samples = min(len(train_data_A), len(train_data_C), len(train_data_F))
    if args.max_samples_per_class:
        min_samples = min(min_samples, args.max_samples_per_class)
    balanced_train_data = (random.sample(train_data_A, min_samples) + 
                           random.sample(train_data_C, min_samples) + 
                           random.sample(train_data_F, min_samples))
    
    log_message(f'Dataset Statistics: Before - A: {len(train_data_A)}, C: {len(train_data_C)}, F: {len(train_data_F)}')
    log_message(f'After - A: {min_samples}, C: {min_samples}, F: {min_samples}, Total: {len(balanced_train_data)}')
    wandb_log({"dataset_size_after_balancing": len(balanced_train_data), "class_samples": min_samples})

    # Create dataset
    train_dataset = SpatialSpectralEEGDataset(data_dir=data_dir, data_info=balanced_train_data, 
                                              scales=scales, time_length=time_length, adjacency_type='combined')

    # Debug batch
    sample_loader = DataLoader(train_dataset, batch_size=4)
    sample_batch, sample_labels = next(iter(sample_loader))
    log_message("\nData Statistics:")
    for key in sample_batch:
        log_message(f"{key} - Shape: {sample_batch[key].shape}, Min: {sample_batch[key].min():.4f}, Max: {sample_batch[key].max():.4f}")
    wandb_log({f"{key}_stats": sample_batch[key].shape[1:] for key in sample_batch})

    # Resume training
    checkpoint_dir = f'{base_dir}/checkpoints'
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
    start_fold, start_epoch = 0, 0
    best_accuracy = 0.0
    if latest_checkpoint:
        fold_num, epoch_num, checkpoint_file = latest_checkpoint
        checkpoint = torch.load(os.path.join(checkpoint_dir, checkpoint_file), map_location=device)
        start_fold, start_epoch = checkpoint['fold'], checkpoint['epoch'] + 1
        best_accuracy = checkpoint.get('best_accuracy', 0.0)
        log_message(f"Resuming from Fold {fold_num}, Epoch {epoch_num}")
    else:
        log_message("Starting from scratch")

    # K-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    contrastive_criterion = nn.CosineEmbeddingLoss(margin=0.5)

    for fold, (train_index, valid_index) in enumerate(skf.split(train_dataset.data, train_dataset.labels)):
        if fold >= args.folds:
            break
        if fold < start_fold:
            continue

        log_message(f'\nFold {fold + 1}/{n_splits}')
        wandb_log({"current_fold": fold + 1})

        # Reset model
        model = create_model(num_channels=num_channels, num_classes=num_classes, dim=dim, scales=scales).to(device)

        # Data loaders
        pin_memory = device.type == 'cuda'
        loader_kwargs = dict(batch_size=batch_size, pin_memory=pin_memory, num_workers=args.num_workers, drop_last=True)
        if args.num_workers > 0:
            loader_kwargs['prefetch_factor'] = 2
        train_dataloader = DataLoader(train_dataset, sampler=SubsetRandomSampler(train_index), **loader_kwargs)
        valid_dataloader = DataLoader(train_dataset, sampler=SubsetRandomSampler(valid_index), **loader_kwargs)

        log_message(f'Train batches: {len(train_dataloader)}, Valid batches: {len(valid_dataloader)}')

        # Optimizer and schedulers
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda e: min(1.0, e / 5.0) if e < 5 else 1.0)
        plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        if fold == start_fold and start_epoch > 0:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
        best_fold_accuracy, best_fold_loss = 0.0, float('inf')
        patience_counter = 0

        # Auxiliary classifiers
        aux_spatial = nn.Linear(dim, num_classes).to(device)
        aux_graph = nn.Linear(dim, num_classes).to(device)

        for epoch in range(start_epoch if fold == start_fold else 0, epochs):
            model.train()
            train_loss, data_time, forward_time, backward_time = 0.0, 0.0, 0.0, 0.0
            optimizer.zero_grad()

            for batch_idx, (inputs_dict, labels) in enumerate(train_dataloader):
                progress_bar(batch_idx + 1, len(train_dataloader), prefix=f'Epoch {epoch + 1}/{epochs}, Batch:')
                data_start = time.time()

                raw_eeg = inputs_dict['raw_eeg'].to(device, non_blocking=True)
                scale_inputs = {k: v.to(device, non_blocking=True) for k, v in inputs_dict.items() if k.startswith('scale_')}
                if 'adjacency' in inputs_dict:
                    scale_inputs['adjacency'] = inputs_dict['adjacency'].to(device, non_blocking=True)
                if 'spatial_positions' in inputs_dict:
                    scale_inputs['spatial_positions'] = inputs_dict['spatial_positions'].to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                data_time += time.time() - data_start
                forward_start = time.time()

                # Forward pass - FIXED: Remove checkpointing that was causing issues
                with (torch.amp.autocast(device_type=device.type) if use_amp else nullcontext()):
                    # Direct forward pass without checkpointing
                    outputs = model(raw_eeg, scale_inputs)

                    # Extract intermediate features (assuming model exposes them)
                    spatial_pooled = model.spatial_extractor(raw_eeg, scale_inputs.get('spatial_positions', None)).mean(0)
                    graph_pooled = model.graph_module(scale_inputs)

                    if graph_pooled.size(1) != dim:
                        graph_pooled = F.pad(graph_pooled, (0, dim - graph_pooled.size(1)))

                    loss_ce = criterion(outputs, labels) / accumulation_steps
                    loss_aux_spatial = criterion(aux_spatial(spatial_pooled), labels) / accumulation_steps
                    loss_aux_graph = criterion(aux_graph(graph_pooled), labels) / accumulation_steps
                    loss_contrastive = contrastive_criterion(spatial_pooled, graph_pooled, 
                                                            torch.ones(batch_size).to(device)) / accumulation_steps
                    loss = loss_ce + 0.3 * (loss_aux_spatial + loss_aux_graph) + 0.1 * loss_contrastive

                forward_time += time.time() - forward_start
                backward_start = time.time()

                scaler.scale(loss).backward()
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    wandb_log({"grad_norm": float(grad_norm)})

                backward_time += time.time() - backward_start
                train_loss += loss.item() * accumulation_steps

            warmup_scheduler.step()
            epoch_train_loss = train_loss / len(train_dataloader)

            # Validation
            model.eval()
            valid_loss, correct, total = 0.0, 0, 0
            all_preds, all_labels = [], []

            with torch.no_grad():
                for inputs_dict, labels in valid_dataloader:
                    raw_eeg = inputs_dict['raw_eeg'].to(device, non_blocking=True)
                    scale_inputs = {k: v.to(device, non_blocking=True) for k, v in inputs_dict.items() if k.startswith('scale_')}
                    if 'adjacency' in inputs_dict:
                        scale_inputs['adjacency'] = inputs_dict['adjacency'].to(device, non_blocking=True)
                    if 'spatial_positions' in inputs_dict:
                        scale_inputs['spatial_positions'] = inputs_dict['spatial_positions'].to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    with (torch.amp.autocast(device_type=device.type) if use_amp else nullcontext()):
                        outputs = model(raw_eeg, scale_inputs)
                        loss = criterion(outputs, labels)
                    valid_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            epoch_valid_loss = valid_loss / len(valid_dataloader)
            accuracy = 100. * correct / total
            precision, recall, f1 = compute_metrics(all_labels, all_preds)

            plateau_scheduler.step(epoch_valid_loss)
            current_lr = optimizer.param_groups[0]['lr']

            # Logging
            log_message(f'Fold {fold + 1}/{n_splits}, Epoch {epoch + 1}/{epochs}, '
                        f'Train Loss: {epoch_train_loss:.4f}, Valid Loss: {epoch_valid_loss:.4f}, '
                        f'Accuracy: {accuracy:.2f}%, F1: {f1:.4f}, LR: {current_lr:.6f}')
            wandb_log({"epoch": epoch + 1, "fold": fold + 1, "train_loss": epoch_train_loss, 
                       "valid_loss": epoch_valid_loss, "accuracy": accuracy, "f1_score": f1, 
                       "learning_rate": current_lr})

            # Checkpoint
            if epoch % 5 == 0 or epoch == epochs - 1:
                torch.save({'epoch': epoch, 'fold': fold, 'model_state_dict': model.state_dict(), 
                            'optimizer_state_dict': optimizer.state_dict(), 'best_accuracy': best_accuracy}, 
                           f'{checkpoint_dir}/fold{fold+1}_epoch{epoch+1}.pt')

            # Early stopping and best model
            if accuracy > best_fold_accuracy:
                best_fold_accuracy = accuracy
                torch.save(model.state_dict(), f'{base_dir}/models/best_model_fold{fold+1}.pth')
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save(model.state_dict(), f'{base_dir}/models/best_model_overall.pth')

            if epoch_valid_loss < best_fold_loss:
                best_fold_loss = epoch_valid_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    log_message(f"Early stopping at epoch {epoch + 1}")
                    break

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    log_message(f"\nDone. Best validation accuracy: {best_accuracy:.2f}%. Models saved in {base_dir}/models")
    if wandb is not None:
        wandb.finish()

if __name__ == "__main__":
    main()