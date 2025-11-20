import argparse
import os, glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from dataset import ImgDataset, train_tf, val_tf
from model import get_model

def main():
    # 1️⃣ Parse Command-Line Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    # 2️⃣ Create Dataset + DataLoaders
    print("Loading dataset...")
    train_path = os.path.join(args.data_dir, "seg_train", "seg_train")
    val_path = os.path.join(args.data_dir, "seg_test", "seg_test")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("ERROR: Dataset folders not found.")
        return

    train_files = glob.glob(os.path.join(train_path, "*/*.jpg"))
    val_files = glob.glob(os.path.join(val_path, "*/*.jpg"))

    if len(train_files) == 0 or len(val_files) == 0:
        print("ERROR: No images found in dataset folders.")
        return

    class_names = sorted(os.listdir(train_path))
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}

    train_labels = [class_to_idx[os.path.basename(os.path.dirname(f))] for f in train_files]
    val_labels   = [class_to_idx[os.path.basename(os.path.dirname(f))] for f in val_files]

    train_dataset = ImgDataset(train_files, train_labels, transforms=train_tf)
    val_dataset   = ImgDataset(val_files, val_labels, transforms=val_tf)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 3️⃣ Create Model + Loss + Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    model = get_model(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 4️⃣ Training Loop
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss, running_correct = 0.0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            running_correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = running_correct / len(train_dataset)

        # Validation
        model.eval()
        val_loss, val_correct = 0.0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                all_preds.extend(outputs.argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = val_correct / len(val_dataset)

        print(f"Epoch [{epoch+1}/{args.epochs}] | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        # Save best checkpoint
        if val_acc > best_acc:
            torch.save(model.state_dict(), "best_model.pth")
            best_acc = val_acc
            print("✅ Saved new best model!")

    # 5️⃣ Final Evaluation
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    main()
