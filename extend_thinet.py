import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
from tqdm import tqdm

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=0, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=0, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 5),
        )
    
    def forward(self, x):
        return self.model(x)

def extend_training():
    print("="*60)
    print("EXTENDED TRAINING ON NUCLEAR MODEL")
    print("="*60)
    
    # Load data
    with open('train_images.pkl', 'rb') as f:
        train_images = pickle.load(f)
    with open('train_labels.pkl', 'rb') as f:
        train_labels = pickle.load(f)
    with open('val_images.pkl', 'rb') as f:
        val_images = pickle.load(f)
    with open('val_labels.pkl', 'rb') as f:
        val_labels = pickle.load(f)
    
    train_images = torch.FloatTensor(train_images).permute(0, 3, 1, 2)
    val_images = torch.FloatTensor(val_images).permute(0, 3, 1, 2)
    train_labels = torch.LongTensor(train_labels.squeeze())
    val_labels = torch.LongTensor(val_labels.squeeze())
    
    train_dataset = TensorDataset(train_images, train_labels)
    val_dataset = TensorDataset(val_images, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    
    # Load nuclear model
    device = torch.device('cuda')
    model = ConvNet().to(device)
    model.load_state_dict(torch.load('nuclear_extended_score0.8270.pt'))
    
    print("Loaded nuclear model (64.36% acc @ 92.36% sparsity)")
    
    # Aggressive fine-tuning
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=2e-3,  # Higher LR for recovery
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True
    )
    
    # Warm restarts for escaping local minima
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )
    
    criterion = nn.CrossEntropyLoss()
    
    best_score = 0
    epochs = 600
    
    print(f"\nFine-tuning for {epochs} epochs with warm restarts...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Maintain sparsity
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if 'weight' in name and param.grad is not None:
                        mask = (param.data != 0).float()
                        param.grad.mul_(mask)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        scheduler.step()
        
        # Evaluate every 25 epochs
        if (epoch + 1) % 25 == 0 or epoch == epochs - 1:
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            val_acc = val_correct / val_total
            
            # Calculate sparsity
            total_zeros = 0
            total_params = 0
            for name, param in model.named_parameters():
                if 'weight' in name:
                    total_zeros += (param.data == 0).sum().item()
                    total_params += param.numel()
            sparsity = total_zeros / total_params
            
            score = (val_acc + sparsity) / 2
            
            print(f"\nEpoch {epoch+1}/{epochs}:")
            print(f"  Val Acc: {val_acc*100:.2f}%")
            print(f"  Sparsity: {sparsity*100:.2f}%")
            print(f"  Score: {score:.6f}")
            print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
            
            if score > best_score:
                best_score = score
                torch.save(model.state_dict(), f'nuclear_super_extended_score{score:.4f}.pt')
                torch.save(model.state_dict(), 'my_model_weights_super_extended.pt',
                          _use_new_zipfile_serialization=False)
                print(f"  ✓ NEW BEST SCORE!")
    
    print(f"\n{'='*60}")
    print(f"EXTENDED TRAINING COMPLETE!")
    print(f"Best score: {best_score:.6f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    extend_training()
