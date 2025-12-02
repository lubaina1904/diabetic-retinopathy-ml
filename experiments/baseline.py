
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from datetime import datetime
import yaml


from src.dataset import DiabeticRetinopathyDataset, get_transforms
from src.model import create_model, count_parameters
from src.train import Trainer, evaluate_model
from src.utils import create_train_val_split, get_dataloader, plot_training_history, plot_confusion_matrix, save_results


def load_config(config_path='../configs/config.yaml'):
    
    config_path = os.path.join(os.path.dirname(__file__), config_path)
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('baseline', {})
    return {}


def main():
    
    config_from_file = load_config()
    

    config = {
        'csv_file': config_from_file.get('csv_file', 'data/aptos/train.csv'),
        'img_dir': config_from_file.get('img_dir', 'data/aptos/train_images'),
        'model_name': config_from_file.get('model_name', 'efficientnet_b0'),
        'num_classes': config_from_file.get('num_classes', 5),
        'pretrained': config_from_file.get('pretrained', True),

        'batch_size': config_from_file.get('batch_size', 32),
        'num_epochs': config_from_file.get('num_epochs', 20),
        'learning_rate': config_from_file.get('learning_rate', 0.001),
        'val_split': config_from_file.get('val_split', 0.2),
        'num_workers': config_from_file.get('num_workers', 2),

        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': config_from_file.get('save_dir', 'results/models/baseline'),
    }

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config['csv_file'] = os.path.join(project_root, config['csv_file'])
    config['img_dir'] = os.path.join(project_root, config['img_dir'])
    config['save_dir'] = os.path.join(project_root, config['save_dir'])

    os.makedirs(config['save_dir'], exist_ok=True)

    print("\nConfiguration:")
    for key, value in config.items():
        print("  {}: {}".format(key, value))

    device = torch.device(config['device'])
    print("\nDevice: {}".format(device))


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

    print("Train batches: {}".format(len(train_loader)))
    print("Val batches: {}".format(len(val_loader)))


    model = create_model(
        model_name=config['model_name'],
        num_classes=config['num_classes'],
        pretrained=config['pretrained']
    )

    count_parameters(model)


    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=config['learning_rate']
    )

    start_time = datetime.now()
    history = trainer.train(
        num_epochs=config['num_epochs'],
        save_dir=config['save_dir']
    )

    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()

    print("\nTraining time: {:.2f} minutes".format(training_time / 60))

    checkpoint = torch.load(
        os.path.join(config['save_dir'], 'best_model.pth'),
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint['model_state_dict'])

    results = evaluate_model(model, val_loader, device)

    figures_dir = os.path.join(project_root, 'results/figures')
    os.makedirs(figures_dir, exist_ok=True)

    plot_training_history(
        history,
        save_path=os.path.join(figures_dir, 'baseline_training_history.png')
    )

    plot_confusion_matrix(
        results['labels'],
        results['predictions'],
        save_path=os.path.join(figures_dir, 'baseline_confusion_matrix.png')
    )

    final_results = {
        'config': config,
        'history': history,
        'evaluation': {
            'accuracy': results['accuracy'],
            'kappa': results['kappa'],
            'confusion_matrix': results['confusion_matrix']
        },
        'training_time_minutes': training_time / 60,
        'best_val_kappa': trainer.best_val_kappa,
        'timestamp': datetime.now().isoformat()
    }

    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    save_results(
        final_results,
        os.path.join(results_dir, 'baseline_results.json')
    )
    print("EXPERIMENT SUMMARY")
    print("Best Validation Accuracy: {:.2f}%".format(results['accuracy']))
    print("Best Validation Kappa: {:.4f}".format(results['kappa']))
    print("Training Time: {:.2f} minutes".format(training_time / 60))
    print("Model saved to: {}".format(config['save_dir']))

    print("\nBaseline experiment complete!")
    print("Check the results folder for:")
    print("  - best_model.pth (trained model)")
    print("  - training_history.png (loss/accuracy curves)")
    print("  - confusion_matrix.png (prediction analysis)")
    print("  - results.json (all metrics)")

    return final_results


if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print("\n\nError occurred: {}".format(e))
        import traceback
        traceback.print_exc()

