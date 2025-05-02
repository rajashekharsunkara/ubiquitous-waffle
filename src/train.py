import os
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from preprocessing import AudioPreprocessor
from model import LungSoundModel, ModelTrainer
from sklearn.model_selection import KFold

def train(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create necessary directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Initialize preprocessing
    preprocessor = AudioPreprocessor(
        sample_rate=22050,
        frame_length=2.5,
        frame_overlap=0.5
    )

    # Load and preprocess dataset
    print("Loading and preprocessing data...")
    features, labels = preprocessor.prepare_dataset(
        args.data_dir,
        args.labels_path
    )

    # Calculate input channels based on feature extraction
    input_channels = features.shape[1]  # Number of feature channels

    # Setup 5-fold cross validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(features)):
        print(f"\nTraining Fold {fold + 1}/5")
        
        # Split data
        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        X_val = torch.FloatTensor(X_val)
        y_train = torch.LongTensor(y_train)
        y_val = torch.LongTensor(y_val)

        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4
        )

        # Initialize model
        model = LungSoundModel(
            input_channels=input_channels,
            num_classes=args.num_classes
        ).to(device)
        
        trainer = ModelTrainer(model, device, learning_rate=args.learning_rate)

        # Training loop
        best_val_loss = float('inf')
        for epoch in range(args.epochs):
            # Train
            train_loss, train_acc = trainer.train_one_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = trainer.validate(val_loader)
            
            print(f'Epoch {epoch+1}/{args.epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save checkpoint if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(
                    args.checkpoint_dir,
                    f'best_model_fold{fold+1}.pth'
                )
                trainer.save_checkpoint(checkpoint_path, epoch, val_loss)
                print(f'Saved checkpoint to {checkpoint_path}')

    print('Training completed!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing the dataset')
    parser.add_argument('--labels_path', type=str, required=True,
                        help='Path to labels CSV file')
    parser.add_argument('--checkpoint_dir', type=str, default='models',
                        help='Directory to save model checkpoints')
    parser.add_argument('--num_classes', type=int, required=True,
                        help='Number of disease classes')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')

    args = parser.parse_args()
    train(args)