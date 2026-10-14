"""
=============================================================================
B.Tech CSE Deep Learning Project: Image Classification
Theme: Fruits Classification (Apple, Banana, Orange, Strawberry)
Framework: PyTorch
=============================================================================
This script handles:
1. Loading the organized Dataset (Training, Validation, Testing splits)
2. Image preprocessing and data augmentation
3. Model definition (Custom Convolutional Neural Network - FruitCNN)
4. Model training with loss and accuracy tracking per epoch
5. Evaluation on Test Set:
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix visualization
   - Training vs Validation Loss & Accuracy curves
   - Sample test image predictions with confidence scores
6. Saving model checkpoints (.pt and .pth) and output visual artifacts
=============================================================================
"""

import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


# =============================================================================
# 1. Configuration & Hyperparameters
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATASET_DIR = os.path.join(PROJECT_ROOT, "Dataset")
MODEL_DIR = os.path.join(PROJECT_ROOT, "Trained_Model")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Sample_Output")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = (100, 100)
BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
RANDOM_SEED = 42

# Reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[*] Execution Device: {DEVICE}")


# =============================================================================
# 2. Image Preprocessing & Data Augmentation
# =============================================================================
# ImageNet normalization statistics
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

data_transforms = {
    'Training': transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ]),
    'Validation': transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ]),
    'Testing': transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ]),
}

print("[*] Loading datasets from:", DATASET_DIR)
image_datasets = {
    split: datasets.ImageFolder(os.path.join(DATASET_DIR, split), transform=data_transforms[split])
    for split in ['Training', 'Validation', 'Testing']
}

dataloaders = {
    'Training': DataLoader(image_datasets['Training'], batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
    'Validation': DataLoader(image_datasets['Validation'], batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
    'Testing': DataLoader(image_datasets['Testing'], batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
}

dataset_sizes = {split: len(image_datasets[split]) for split in ['Training', 'Validation', 'Testing']}
class_names = image_datasets['Training'].classes
num_classes = len(class_names)

print(f"[*] Classes ({num_classes}): {class_names}")
print(f"[*] Dataset Splits: Training={dataset_sizes['Training']}, Validation={dataset_sizes['Validation']}, Testing={dataset_sizes['Testing']}")


# =============================================================================
# 3. Model Architecture (Custom FruitCNN)
# =============================================================================
class FruitCNN(nn.Module):
    """
    Convolutional Neural Network tailored for Multi-Class Fruit Image Classification.
    Features:
    - 3 Convolutional Blocks with Batch Normalization and ReLU
    - Max Pooling for spatial dimension reduction
    - Adaptive Average Pooling for size-invariance
    - Fully Connected Classification layers with Dropout for regularization
    """
    def __init__(self, num_classes=4):
        super(FruitCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Conv Block 1: Input (3, 100, 100) -> Output (32, 50, 50)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 2: Input (32, 50, 50) -> Output (64, 25, 25)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 3: Input (64, 25, 25) -> Output (128, 12, 12)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),  # Feature map: 128 x 4 x 4 = 2048
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = FruitCNN(num_classes=num_classes).to(DEVICE)
print("\n" + "="*50)
print("Model Architecture Summary:")
print(model)
print("="*50 + "\n")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.5)


# =============================================================================
# 4. Training and Validation Loop
# =============================================================================
def train_model(model, criterion, optimizer, scheduler, num_epochs=15):
    since = time.time()
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    print("="*65)
    print(f"{'Epoch':^8} | {'Train Loss':^12} | {'Train Acc':^10} | {'Val Loss':^10} | {'Val Acc':^10}")
    print("="*65)
    
    for epoch in range(1, num_epochs + 1):
        # Each epoch has a training and validation phase
        for phase in ['Training', 'Validation']:
            if phase == 'Training':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'Training'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'Training':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            if phase == 'Training' and scheduler is not None:
                scheduler.step()
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = (running_corrects.double() / dataset_sizes[phase]).item()
            
            if phase == 'Training':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)
                
                # Check for best model
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    
        print(f"{epoch:^8} | {history['train_loss'][-1]:^12.4f} | {history['train_acc'][-1]*100:^9.2f}% | {history['val_loss'][-1]:^10.4f} | {history['val_acc'][-1]*100:^9.2f}%")
        
    time_elapsed = time.time() - since
    print("="*65)
    print(f"[*] Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"[*] Best Validation Accuracy: {best_acc * 100:.2f}%\n")
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history


print("[*] Starting Model Training...")
trained_model, training_history = train_model(
    model, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS
)


# =============================================================================
# 5. Saving Trained Model
# =============================================================================
weights_path = os.path.join(MODEL_DIR, "fruit_classifier_weights.pth")
full_model_path = os.path.join(MODEL_DIR, "fruit_classifier_model.pt")

torch.save(trained_model.state_dict(), weights_path)
torch.save(trained_model, full_model_path)
print(f"[+] Model weights saved to: {weights_path}")
print(f"[+] Full model saved to: {full_model_path}")


# =============================================================================
# 6. Evaluation on Test Set
# =============================================================================
print("\n[*] Evaluating model on Independent Test Set...")
trained_model.eval()

y_true = []
y_pred = []
y_probs = []
test_images_raw = []

with torch.no_grad():
    for inputs, labels in dataloaders['Testing']:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        
        outputs = trained_model(inputs)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_probs.extend(probs.cpu().numpy())
        test_images_raw.extend(inputs.cpu())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_probs = np.array(y_probs)

test_acc = accuracy_score(y_true, y_pred)
test_precision_macro = precision_score(y_true, y_pred, average='macro')
test_recall_macro = recall_score(y_true, y_pred, average='macro')
test_f1_macro = f1_score(y_true, y_pred, average='macro')

test_precision_weighted = precision_score(y_true, y_pred, average='weighted')
test_recall_weighted = recall_score(y_true, y_pred, average='weighted')
test_f1_weighted = f1_score(y_true, y_pred, average='weighted')

report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
conf_matrix = confusion_matrix(y_true, y_pred)

print("\n" + "="*50)
print(f"Test Accuracy:           {test_acc * 100:.2f}%")
print(f"Test Precision (Macro):  {test_precision_macro * 100:.2f}%")
print(f"Test Recall (Macro):     {test_recall_macro * 100:.2f}%")
print(f"Test F1-Score (Macro):   {test_f1_macro * 100:.2f}%")
print("="*50)
print("\nClassification Report:\n")
print(report)

# Save text metrics report
metrics_file = os.path.join(OUTPUT_DIR, "classification_metrics.txt")
with open(metrics_file, "w") as f:
    f.write("="*60 + "\n")
    f.write("      DEEP LEARNING MODEL PERFORMANCE EVALUATION REPORT\n")
    f.write("="*60 + "\n\n")
    f.write(f"Dataset Theme: Fruits Classification\n")
    f.write(f"Target Classes ({len(class_names)}): {', '.join(class_names)}\n")
    f.write(f"Total Test Samples: {len(y_true)}\n\n")
    f.write("KEY METRICS:\n")
    f.write(f"- Accuracy:            {test_acc * 100:.2f}%\n")
    f.write(f"- Precision (Macro):   {test_precision_macro * 100:.2f}%\n")
    f.write(f"- Recall (Macro):      {test_recall_macro * 100:.2f}%\n")
    f.write(f"- F1-Score (Macro):    {test_f1_macro * 100:.2f}%\n\n")
    f.write(f"- Precision (Weighted):{test_precision_weighted * 100:.2f}%\n")
    f.write(f"- Recall (Weighted):   {test_recall_weighted * 100:.2f}%\n")
    f.write(f"- F1-Score (Weighted): {test_f1_weighted * 100:.2f}%\n\n")
    f.write("DETAILED CLASSIFICATION REPORT:\n")
    f.write(report + "\n\n")
    f.write("CONFUSION MATRIX:\n")
    f.write(str(conf_matrix) + "\n")

print(f"[+] Detailed metrics report saved to: {metrics_file}")


# =============================================================================
# 7. Visualization Artifacts Generation
# =============================================================================

# 7.1 Accuracy & Loss Curves
epochs_range = range(1, NUM_EPOCHS + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Deep Learning Model Training & Validation Performance', fontsize=15, fontweight='bold', y=1.02)

# Loss Plot
ax1.plot(epochs_range, training_history['train_loss'], 'o-', color='#1f77b4', linewidth=2.2, label='Training Loss')
ax1.plot(epochs_range, training_history['val_loss'], 's--', color='#ff7f0e', linewidth=2.2, label='Validation Loss')
ax1.set_title('Cross-Entropy Loss vs. Epochs', fontsize=13, fontweight='semibold')
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Loss', fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='upper right', frameon=True)

# Accuracy Plot
train_acc_pct = [acc * 100 for acc in training_history['train_acc']]
val_acc_pct = [acc * 100 for acc in training_history['val_acc']]
ax2.plot(epochs_range, train_acc_pct, 'o-', color='#2ca02c', linewidth=2.2, label='Training Accuracy')
ax2.plot(epochs_range, val_acc_pct, 's--', color='#d62728', linewidth=2.2, label='Validation Accuracy')
ax2.set_title('Classification Accuracy (%) vs. Epochs', fontsize=13, fontweight='semibold')
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Accuracy (%)', fontsize=11)
ax2.set_ylim([0, 105])
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='lower right', frameon=True)

plt.tight_layout()
loss_acc_path = os.path.join(OUTPUT_DIR, "accuracy_loss_curves.png")
plt.savefig(loss_acc_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"[+] Saved Accuracy and Loss graph to: {loss_acc_path}")


# 7.2 Confusion Matrix Heatmap
plt.figure(figsize=(7, 6))
sns.set_theme(style='white')
ax = sns.heatmap(
    conf_matrix, 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    cbar=True,
    xticklabels=class_names, 
    yticklabels=class_names,
    annot_kws={'size': 14, 'fontweight': 'bold'}
)
plt.title(f'Confusion Matrix (Test Accuracy: {test_acc * 100:.1f}%)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Predicted Class', fontsize=12, fontweight='semibold')
plt.ylabel('Ground Truth Class', fontsize=12, fontweight='semibold')
plt.tight_layout()

conf_matrix_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(conf_matrix_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"[+] Saved Confusion Matrix to: {conf_matrix_path}")


# 7.3 Sample Prediction Results Grid
def denormalize_image(tensor):
    """Denormalizes a PyTorch tensor for matplotlib rendering."""
    tensor = tensor.clone().detach().cpu()
    for t, m, s in zip(tensor, NORM_MEAN, NORM_STD):
        t.mul_(s).add_(m)
    tensor = torch.clamp(tensor, 0, 1)
    return tensor.numpy().transpose((1, 2, 0))

indices = np.random.choice(len(y_true), size=min(12, len(y_true)), replace=False)
rows, cols = 3, 4
fig, axes = plt.subplots(rows, cols, figsize=(14, 11))
fig.suptitle('Sample Test Predictions with Confidence Scores', fontsize=16, fontweight='bold', y=0.99)

for i, idx in enumerate(indices):
    r = i // cols
    c = i % cols
    ax = axes[r, c]
    
    img = denormalize_image(test_images_raw[idx])
    true_label = class_names[y_true[idx]]
    pred_label = class_names[y_pred[idx]]
    confidence = y_probs[idx][y_pred[idx]] * 100
    
    is_correct = (y_true[idx] == y_pred[idx])
    title_color = '#1b7837' if is_correct else '#b2182b'
    status_icon = '[CORRECT]' if is_correct else '[INCORRECT]'
    
    ax.imshow(img)
    ax.set_title(
        f"{status_icon} Pred: {pred_label} ({confidence:.1f}%)\nTrue: {true_label}",
        fontsize=10,
        fontweight='bold',
        color=title_color,
        pad=6
    )
    ax.axis('off')

# Hide any empty axes if len(indices) < rows*cols
for i in range(len(indices), rows * cols):
    r = i // cols
    c = i % cols
    axes[r, c].axis('off')

plt.tight_layout()
sample_preds_path = os.path.join(OUTPUT_DIR, "sample_predictions.png")
plt.savefig(sample_preds_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"[+] Saved Sample Predictions to: {sample_preds_path}")

print("\n" + "="*50)
print("[SUCCESS] ALL TRAINING AND EVALUATION TASKS COMPLETED!")
print("="*50)
