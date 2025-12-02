import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
from tqdm import tqdm

# ============================================
# MODEL ARCHITECTURE (from starter code)
# ============================================
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=True),      # 0
            nn.ReLU(),                                                    # 1
            nn.Conv2d(32, 32, kernel_size=3, padding=0, bias=True),      # 2
            nn.ReLU(),                                                    # 3
            nn.MaxPool2d(kernel_size=2, stride=2),                       # 4
            nn.Dropout(0.25),                                             # 5
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True),      # 6
            nn.ReLU(),                                                    # 7
            nn.Conv2d(64, 64, kernel_size=3, padding=0, bias=True),      # 8
            nn.ReLU(),                                                    # 9
            nn.MaxPool2d(kernel_size=2, stride=2),                       # 10
            nn.Dropout(0.25),                                             # 11
            nn.Flatten(),                                                 # 12
            nn.Linear(1024, 512),                                         # 13
            nn.ReLU(),                                                    # 14
            nn.Dropout(0.5),                                              # 15
            nn.Linear(512, 5),                                            # 16
        )
    
    def forward(self, x):
        return self.model(x)

# ============================================
# FAST THINET PRUNER
# ============================================
class FastThiNetPruner:
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
    def collect_activation_importance(self, num_batches=50):
        """
        Collect activation magnitudes as proxy for channel importance.
        Fast alternative to reconstruction error calculation.
        """
        self.model.eval()
        
        # Track which conv layers to analyze
        conv_layers = [0, 2, 6, 8]  # indices of Conv2d layers
        
        # Store activation statistics
        activation_stats = {idx: [] for idx in conv_layers}
        
        # Hook to collect activations
        hooks = []
        activations = {}
        
        def get_activation(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
        
        # Register hooks
        for idx in conv_layers:
            hook = self.model.model[idx].register_forward_hook(get_activation(idx))
            hooks.append(hook)
        
        print("Collecting activation statistics...")
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(self.train_loader):
                if batch_idx >= num_batches:
                    break
                
                inputs = inputs.to(self.device)
                _ = self.model(inputs)
                
                # Collect channel-wise statistics
                for idx in conv_layers:
                    act = activations[idx]  # [B, C, H, W]
                    # Average absolute activation per channel
                    channel_importance = act.abs().mean(dim=[0, 2, 3])  # [C]
                    activation_stats[idx].append(channel_importance.cpu())
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Average across batches
        for idx in conv_layers:
            activation_stats[idx] = torch.stack(activation_stats[idx]).mean(dim=0)
        
        print("Activation statistics collected!")
        return activation_stats
    
    def prune_channels_by_importance(self, activation_stats, target_sparsity=0.93):
        """
        Prune channels based on activation importance.
        Uses layer-wise adaptive pruning ratios.
        """
        print(f"\n{'='*60}")
        print(f"PRUNING TO TARGET {target_sparsity*100:.1f}% SPARSITY")
        print(f"{'='*60}")
        
        # Layer-wise pruning ratios (inspired by ThiNet paper)
        # Early layers: more aggressive (less important for final accuracy)
        # Late layers: conservative (more critical)
        # pruning_ratios = {
        #     0: 0.30,   # First conv: prune 30% of channels
        #     2: 0.35,   # Second conv: prune 35%
        #     6: 0.25,   # Third conv: prune 25% (entering deeper layers)
        #     8: 0.20,   # Fourth conv: prune 20% (most conservative)
        # }

        # new 
        # pruning_ratios = {
        #     0: 0.60,   # prune 60% of channels
        #     2: 0.65,   # prune 65%
        #     6: 0.50,   # prune 50%
        #     8: 0.45,   # prune 45%
        # }

        # Make pruning SLIGHTLY less aggressive on conv layers
        pruning_ratios = {0: 0.45, 2: 0.50, 6: 0.35, 8: 0.30}
                
        total_pruned = 0
        total_params = 0
        
        for layer_idx, importance in activation_stats.items():
            prune_ratio = pruning_ratios[layer_idx]
            num_channels = len(importance)
            num_keep = int(num_channels * (1 - prune_ratio))
            
            # Select top-k channels by importance
            _, top_indices = torch.topk(importance, num_keep)
            
            # Create channel mask
            mask = torch.zeros(num_channels)
            mask[top_indices] = 1.0
            
            # Apply mask to conv layer weights
            conv_layer = self.model.model[layer_idx]
            with torch.no_grad():
                # Zero out pruned channels
                for c in range(num_channels):
                    if mask[c] == 0:
                        conv_layer.weight.data[c, :, :, :] = 0
                
                # Also zero out corresponding input channels in next conv
                # (structured pruning propagation)
                if layer_idx in [0, 6]:  # First conv in each block
                    next_conv_idx = layer_idx + 2
                    next_conv = self.model.model[next_conv_idx]
                    for c in range(num_channels):
                        if mask[c] == 0:
                            next_conv.weight.data[:, c, :, :] = 0
            
            # Calculate pruned params
            layer_params = conv_layer.weight.numel()
            layer_pruned = (mask == 0).sum().item() * (layer_params // num_channels)
            total_pruned += layer_pruned
            total_params += layer_params
            
            print(f"Layer {layer_idx}: Kept {num_keep}/{num_channels} channels "
                  f"({(1-prune_ratio)*100:.1f}% kept)")
        
        # Also apply magnitude pruning to FC layers for additional sparsity
        print("\nApplying magnitude pruning to FC layers...")
        fc_layers = [13, 16]
        for idx in fc_layers:
            fc_layer = self.model.model[idx]
            weights = fc_layer.weight.data
            
            # More conservative for FC (they're critical)
            # fc_sparsity = 0.80 if idx == 13 else 0.50  # 80% for first FC, 50% for output
            fc_sparsity = 0.90 if idx == 13 else 0.75
            threshold = torch.quantile(weights.abs().flatten(), fc_sparsity)
            
            mask = (weights.abs() > threshold).float()
            weights.mul_(mask)
            
            fc_pruned = (mask == 0).sum().item()
            fc_params = weights.numel()
            total_pruned += fc_pruned
            total_params += fc_params
            
            print(f"Layer {idx} (FC): {fc_sparsity*100:.1f}% sparsity "
                  f"({fc_pruned}/{fc_params} pruned)")
        
        global_sparsity = total_pruned / total_params
        print(f"\n{'='*60}")
        print(f"GLOBAL SPARSITY ACHIEVED: {global_sparsity*100:.2f}%")
        print(f"{'='*60}\n")
        
        return global_sparsity
    
    def evaluate(self):
        """Evaluate model accuracy"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return correct / total
    
    def calculate_sparsity(self):
        """Calculate global sparsity"""
        total_zeros = 0
        total_params = 0
        
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                total_zeros += (param.data == 0).sum().item()
                total_params += param.numel()
        
        return total_zeros / total_params
    
    def fine_tune(self, epochs=100, lr=1e-3):
        """
        Aggressive fine-tuning with smart scheduling.
        This is the KEY to recovering accuracy!
        """
        print(f"\n{'='*60}")
        print(f"FINE-TUNING FOR {epochs} EPOCHS")
        print(f"{'='*60}\n")
        
        # Optimizer with momentum (better for pruned networks)
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=1e-4,
            nesterov=True
        )
        
        # Cosine annealing scheduler (smooth LR decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
        
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0
        best_score = 0
        best_epoch = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                
                # CRITICAL: Zero out gradients for pruned weights
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if 'weight' in name and param.grad is not None:
                            mask = (param.data != 0).float()
                            param.grad.mul_(mask)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                pbar.set_postfix({'loss': train_loss/(pbar.n+1), 
                                 'acc': 100.*correct/total})
            
            scheduler.step()
            
            # Evaluate every 5 epochs
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                val_acc = self.evaluate()
                sparsity = self.calculate_sparsity()
                score = (val_acc + sparsity) / 2
                
                print(f"\nEpoch {epoch+1}/{epochs}:")
                print(f"  Train Acc: {100.*correct/total:.2f}%")
                print(f"  Val Acc: {val_acc*100:.2f}%")
                print(f"  Sparsity: {sparsity*100:.2f}%")
                print(f"  Score: {score:.6f}")
                print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
                
                if score > best_score:
                    best_score = score
                    best_acc = val_acc
                    best_epoch = epoch + 1
                    # Save best model
                    torch.save(self.model.state_dict(), 
                              f'thinet_best_score{score:.4f}.pt')
                    print(f"  ✓ NEW BEST SCORE!")
        
        print(f"\n{'='*60}")
        print(f"FINE-TUNING COMPLETE!")
        print(f"Best Score: {best_score:.6f} (Epoch {best_epoch})")
        print(f"Best Accuracy: {best_acc*100:.2f}%")
        print(f"{'='*60}\n")
        
        return best_score, best_acc

# ============================================
# MAIN EXECUTION
# ============================================
def main():
    print("="*60)
    print("FAST THINET PRUNER")
    print("="*60)
    
    # Load data
    print("\nLoading dataset...")
    with open('train_images.pkl', 'rb') as f:
        train_images = pickle.load(f)
    with open('train_labels.pkl', 'rb') as f:
        train_labels = pickle.load(f)
    with open('val_images.pkl', 'rb') as f:
        val_images = pickle.load(f)
    with open('val_labels.pkl', 'rb') as f:
        val_labels = pickle.load(f)
    
    # Convert to tensors
    train_images = torch.FloatTensor(train_images).permute(0, 3, 1, 2)
    val_images = torch.FloatTensor(val_images).permute(0, 3, 1, 2)
    train_labels = torch.LongTensor(train_labels.squeeze())
    val_labels = torch.LongTensor(val_labels.squeeze())
    
    # Create dataloaders
    train_dataset = TensorDataset(train_images, train_labels)
    val_dataset = TensorDataset(val_images, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Load pretrained model
    print("\nLoading pretrained model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvNet()
    model.load_state_dict(torch.load('baseline_improved.pt'))
    model = model.to(device)
    
    # Evaluate baseline
    pruner = FastThiNetPruner(model, train_loader, val_loader, device)
    baseline_acc = pruner.evaluate()
    print(f"Baseline accuracy: {baseline_acc*100:.2f}%")
    
    # Step 1: Collect activation importance
    activation_stats = pruner.collect_activation_importance(num_batches=50)
    
    # Step 2: Prune channels
    sparsity = pruner.prune_channels_by_importance(activation_stats, target_sparsity=0.93)
    
    # Evaluate after pruning
    pruned_acc = pruner.evaluate()
    print(f"Accuracy after pruning: {pruned_acc*100:.2f}%")
    print(f"Sparsity: {sparsity*100:.2f}%")
    print(f"Initial score: {(pruned_acc + sparsity)/2:.6f}")
    
    # Step 3: Fine-tune
    # best_score, best_acc = pruner.fine_tune(epochs=100, lr=1e-3)
    best_score, best_acc = pruner.fine_tune(epochs=300, lr=8e-4)

    
    # Final evaluation
    final_acc = pruner.evaluate()
    final_sparsity = pruner.calculate_sparsity()
    final_score = (final_acc + final_sparsity) / 2
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Baseline: {baseline_acc*100:.2f}% accuracy")
    print(f"Final: {final_acc*100:.2f}% accuracy @ {final_sparsity*100:.2f}% sparsity")
    print(f"Score: {final_score:.6f}")
    print(f"Score improvement: {final_score - (baseline_acc + 0)/2:.6f}")
    
    # Save final model
    torch.save(model.state_dict(), 'thinet_from_better_baseline_final.pt')
    torch.save(model.state_dict(), 'thinet_from_better_baseline.pt',
               _use_new_zipfile_serialization=False)
    
    print("\nModels saved:")
    print("  - thinet_final.pt")
    print("  - thinet_from_better_baseline.pt (submission format)")
    print("="*60)

if __name__ == "__main__":
    main()
