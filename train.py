import os
import torch
import models
import argparse
import pandas as pd
from torch import nn
from tqdm import tqdm
from torch.optim import lr_scheduler
from dataset import get_train_data_loader, get_val_test_data_loader
from visualize import store_samples, store_accuracy_loss_graphs, store_classification_plot
from utils import get_train_val_test_split, extract_frames_and_create_csv, get_classification_data

def get_args():
    parser = argparse.ArgumentParser(description='Liveness Detection - Zalo AI Challenge Training')
    parser.add_argument('--generate_image_data', action=argparse.BooleanOptionalAction)
    parser.add_argument('--lr', default=3e-5, type=float, help="Learning rate to train model")
    parser.add_argument('--epochs', default=10, type=int, help="Number of training epochs e.g 25")
    parser.add_argument('--batch_size', default=32, type=int, help="Number of images per batch e.g. 256")
    args = parser.parse_args()
    return args

def GetCorrectPredCount(pPrediction, pLabels):
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def _train(model, device, train_loader, optimizer, scheduler, criterion, train_losses, train_acc):
    model.train()
    
    pbar = tqdm(train_loader)

    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        pred = model(data)

        loss = criterion(pred, target)
        train_loss+=loss.item()

        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler != None:
            scheduler.step()

        correct += GetCorrectPredCount(pred, target)
        processed += len(data)

        pbar.set_description(desc= f'Train: Batch_id={batch_idx} | Loss={loss.item():0.4f} | Accuracy={100*correct/processed:0.2f} | Norm: {norm:.2f}')

    train_acc.append(100*correct/processed)
    train_losses.append(train_loss/len(train_loader))

def _test(model, device, test_loader, criterion, test_losses, test_acc):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device) 

            output = model(data) 
            test_loss += criterion(output, target, reduction='sum').item() 

            correct += GetCorrectPredCount(output, target)

    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
def start_training(num_epochs, model, device, train_loader, test_loader, optimizer, criterion, scheduler):
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []

    for epoch in range(1, num_epochs+1):
        print(f'Epoch {epoch}')
        if scheduler != None:
            print(f'Current learning rate: {scheduler.get_last_lr()}')
        _train(model, device, train_loader, optimizer, scheduler, criterion, train_losses, train_acc)
        _test(model, device, test_loader, criterion, test_losses, test_acc)

    return train_losses, train_acc, test_losses, test_acc

def main():
    args = get_args()

    os.makedirs('images', exist_ok=True)

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    data_df = pd.read_csv('data/train/train/label.csv')
    train_df, val_df, test_df = get_train_val_test_split(data_df)

    if args.generate_image_data:
        extract_frames_and_create_csv('data/train/train/videos', train_df, "data/train", num_samples=10)
        extract_frames_and_create_csv('data/train/train/videos', val_df, "data/val", num_samples=10)
        extract_frames_and_create_csv('data/train/train/videos', test_df, "data/test", num_samples=10)

    dataloader_args = dict(shuffle=True, batch_size=args.batch_size, num_workers=4,
                           pin_memory=True) if cuda else dict(shuffle=True, batch_size=16)
    
    train_images_df = pd.read_csv('data/train/labels.csv')
    val_images_df = pd.read_csv('data/val/labels.csv')
    test_images_df = pd.read_csv('data/test/labels.csv')
    
    train_loader = get_train_data_loader("data/train", train_images_df, **dataloader_args)
    val_loader = get_val_test_data_loader("data/val", val_images_df, **dataloader_args)
    test_loader = get_val_test_data_loader("data/test", test_images_df, **dataloader_args)

    store_samples(train_loader, 'images/augmentation.png', number_of_images=20)

    model = models.ResNet18(num_classes=2)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler_linear = lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=10)
    scheduler_cosine = lr_scheduler.CosineAnnealingLR(optimizer, T_max=490, eta_min=args.lr/100)
    scheduler_lr = lr_scheduler.SequentialLR(optimizer, [scheduler_linear,scheduler_cosine],milestones=[10])

    train_losses, train_acc, test_losses, test_acc = start_training(
        args.epochs, model, device, train_loader, test_loader, optimizer, criterion,
        scheduler_lr
    )

    store_accuracy_loss_graphs(train_losses, train_acc, test_losses, test_acc, 'images/metrics.png')

    classification_data = get_classification_data(model, device, test_loader)
    store_classification_plot(classification_data, 'images/results.png', number_of_samples=20)

    torch.save(model.state_dict(), 'model.pth')


if __name__ == "__main__":
    main()