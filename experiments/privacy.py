
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
import yaml

from src.dataset import DiabeticRetinopathyDataset, get_transforms
from src.model import create_model, count_parameters
from src.privacy import make_private, get_privacy_spent, create_dp_optimizer
from src.utils import create_train_val_split, get_dataloader, save_results
from tqdm import tqdm


def load_config(config_path='../configs/config.yaml'):
    config_path = os.path.join(os.path.dirname(__file__), config_path)
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('privacy', {})
    return {}


def train_dp_model(model, train_loader, val_loader, device, num_epochs, 
                   max_grad_norm=1.0, target_epsilon=10.0):
 
    optimizer = create_dp_optimizer(model, learning_rate=0.001)
    
    privacy_engine, model, optimizer = make_private(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        max_grad_norm=max_grad_norm,
        target_epsilon=target_epsilon,
        epochs=num_epochs
    )
    
    criterion = nn.CrossEntropyLoss()
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epsilon': []
    }
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        epsilon = get_privacy_spent(privacy_engine)
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        history['epsilon'].append(epsilon)
        
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Epsilon: {epsilon:.2f}")
    
    return history, privacy_engine


def main():
    
    config_from_file = load_config()
    
    config = {
        'csv_file': config_from_file.get('csv_file', 'data/aptos/train.csv'),
        'img_dir': config_from_file.get('img_dir', 'data/aptos/train_images'),
        'model_name': config_from_file.get('model_name', 'efficientnet_b0'),
        'num_classes': config_from_file.get('num_classes', 5),
        'pretrained': config_from_file.get('pretrained', True),
        'batch_size': config_from_file.get('batch_size', 32),
        'num_epochs': config_from_file.get('num_epochs', 10),
        'val_split': config_from_file.get('val_split', 0.2),
        'num_workers': config_from_file.get('num_workers', 2),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': config_from_file.get('save_dir', 'results/models/dp'),
        'max_grad_norm': config_from_file.get('max_grad_norm', 1.0),
        'target_epsilon': config_from_file.get('target_epsilon', 10.0),
    }

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config['csv_file'] = os.path.join(project_root, config['csv_file'])
    config['img_dir'] = os.path.join(project_root, config['img_dir'])
    config['save_dir'] = os.path.join(project_root, config['save_dir'])

    os.makedirs(config['save_dir'], exist_ok=True)

    device = torch.device(config['device'])

    print("\nConfiguration:")
    for key, value in config.items():
        print("  {}: {}".format(key, value))

    train_transform = get_transforms(mode='train')
    val_transform = get_transforms(mode='val')

    full_dataset = DiabeticRetinopathyDataset(
        csv_file=config['csv_file'],
        img_dir=config['img_dir'],
        transform=train_transform
    )

    train_dataset, val_dataset = create_train_val_split(
        full_dataset,
        val_split=config['val_split'],
        random_seed=42
    )

    val_dataset.dataset.transform = val_transform

    train_loader = get_dataloader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )

    val_loader = get_dataloader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    model = create_model(
        model_name=config['model_name'],
        num_classes=config['num_classes'],
        pretrained=config['pretrained']
    )

    start_time = datetime.now()
    history, privacy_engine = train_dp_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=config['num_epochs'],
        max_grad_norm=config['max_grad_norm'],
        target_epsilon=config['target_epsilon']
    )
    end_time = datetime.now()

    final_epsilon = get_privacy_spent(privacy_engine)
    
    print(f"Final Privacy Budget (epsilon): {final_epsilon:.2f}")
    print(f"Target epsilon: {config['target_epsilon']}")
    print(f"Training time: {(end_time - start_time).total_seconds() / 60:.2f} minutes")

    results = {
        'config': config,
        'history': history,
        'final_epsilon': final_epsilon,
        'timestamp': datetime.now().isoformat()
    }

    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    save_results(results, os.path.join(results_dir, 'dp_results.json'))
    
    print("\nPrivacy experiment complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print("\n\nError occurred: {}".format(e))
        import traceback
        traceback.print_exc()

