import torch
import pandas as pd

from utils.plotting import plot_training
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
def train_one_epoch(model, loader, criterion, optimizer, device, model_name):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        if model_name == "ffnn":
            inputs = inputs.view(inputs.size(0), -1).to(device)
        elif model_name == "lstm_fcn" or model_name == "conv":
            inputs = inputs.to(device)
        else:
            inputs = inputs.permute(0, 2, 1).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.argmax(dim=1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels.argmax(dim=1)).sum().item()

    train_loss = running_loss / len(loader.dataset)
    train_acc = correct / total
    
    return train_loss, train_acc

def validate_train(model, test_loader, criterion, device, model_name):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            if model_name == "ffnn":
                inputs = inputs.view(inputs.size(0), -1).to(device)
            elif model_name == "lstm_fcn" or model_name == "conv":
                inputs = inputs.to(device)
            else:
                inputs = inputs.permute(0, 2, 1).to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels.argmax(dim=1))
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels.argmax(dim=1)).sum().item()

        test_loss = val_loss / len(test_loader.dataset)
        test_accuracy = correct / total

        print(f"Pérdida de test: {test_loss}")
        print(f"Precisión de test: {test_accuracy}")

        return test_loss, test_accuracy


def validate_real_data(model, real_test_loader, device, model_name):
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in real_test_loader:
            if model_name == "ffnn":
                    inputs = inputs.view(inputs.size(0), -1).to(device)
            elif model_name == "lstm_fcn" or model_name == "conv":
                inputs = inputs.to(device)
            else:
                inputs = inputs.permute(0, 2, 1).to(device)   
            labels = labels.to(device)  
            outputs = model(inputs)
            probs = torch.exp(outputs)[:, 1].detach().cpu().numpy()  
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()  
            true_labels = torch.argmax(labels, dim=1).detach().cpu().numpy()  

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(true_labels)

        return all_probs, all_preds, all_labels


def evaluate_metrics(model, inputs, labels, device, model_name):
    model.eval()
    with torch.no_grad():
        
        if model_name == "ffnn":
            inputs = inputs.view(inputs.size(0), -1).to(device)
        elif model_name == "lstm_fcn" or model_name == "conv":
            inputs = inputs.to(device)
        else:
            inputs = inputs.permute(0, 2, 1).to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        true = torch.argmax(labels, dim=1).cpu().numpy()

    precision = precision_score(true, preds, average="macro")
    recall = recall_score(true, preds, average="macro")
    f1 = f1_score(true, preds, average="macro")
    acc = accuracy_score(true, preds)

    return acc, precision, recall, f1    
    

def train_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, device, model_name, 
               epochs=80, patience=10, save_path='best_model.pth', plot_path='plot.png'):
    best_val_loss = float('inf')
    no_improve = 0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, model_name)
        val_loss, val_acc = validate_train(model, val_loader, criterion, device, model_name)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), save_path) #mejorar
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered.")
                break
    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_acc": train_accuracies,
        "val_acc": val_accuracies
    }

    plot_training(history, plot_path)

    return model, history
