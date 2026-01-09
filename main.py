import torch
from torch.utils.data import DataLoader
from models.deeplob import DeepLOB
from data.dataset import LOBDataset
from training.train import train, evaluate
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Train DeepLOB model')
    parser.add_argument('--data_dir', type=str, 
                        default='data/BenchmarkDatasets/NoAuction/3.NoAuction_DecPre',
                        help='Path to dataset directory')
    parser.add_argument('--k', type=int, default=10, 
                        help='Prediction horizon (10, 20, 50, 100)')
    parser.add_argument('--epochs', type=int, default=50, 
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, 
                        help='Learning rate')
    parser.add_argument('--T', type=int, default=100, 
                        help='History window size')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_data_dir = os.path.join(args.data_dir, 'NoAuction_DecPre_Training')
    test_data_dir = os.path.join(args.data_dir, 'NoAuction_DecPre_Testing')
    
    print("Loading training data...")
    train_dataset = LOBDataset(train_data_dir, train=True, k=args.k, T=args.T)
    print(f"Training samples: {len(train_dataset)}")
    
    print("Loading test data...")
    test_dataset = LOBDataset(test_data_dir, train=False, k=args.k, T=args.T)
    print(f"Test samples: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True	)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = DeepLOB(y_len=3)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    history = train(model, train_loader, test_loader, epochs=args.epochs, lr=args.lr, device=device)
    
    # Final evaluation
    print("\nFinal Evaluation:")
    criterion = torch.nn.CrossEntropyLoss()
    final_metrics = evaluate(model, test_loader, criterion, device)
    print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"  Precision: {final_metrics['precision']:.4f}")
    print(f"  Recall: {final_metrics['recall']:.4f}")
    print(f"  F1 Score: {final_metrics['f1']:.4f}")
    
    # Save model
    torch.save(model.state_dict(), 'deeplob_model.pth')
    print("Model saved to deeplob_model.pth")

if __name__ == '__main__':
    main()
