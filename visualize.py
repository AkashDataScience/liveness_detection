import math
import numpy as np
import albumentations
import matplotlib.pyplot as plt

CLASS_NAMES= ['Fake', 'Real']

def get_inv_transforms():
    inv_transforms = albumentations.Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                                              [1/0.229, 1/0.224, 1/0.225], max_pixel_value=1.0)
    return inv_transforms

def store_samples(train_loader, path, number_of_images=20):
    inv_transform = get_inv_transforms()

    figure = plt.figure()
    x_count = 5
    y_count = 1 if number_of_images <= 5 else math.floor(number_of_images / x_count)
    images, labels = next(iter(train_loader))

    for index in range(1, number_of_images + 1):
        plt.subplot(y_count, x_count, index)
        plt.title(CLASS_NAMES[labels[index].numpy()])
        plt.axis('off')
        image = np.array(images[index])
        image = np.transpose(image, (1, 2, 0))
        image = inv_transform(image=image)['image']
        plt.imshow(image)

    plt.tight_layout()
    plt.savefig(path)
    print(f"Sample augmented images are saved at {path}")

def store_accuracy_loss_graphs(train_losses, train_acc, test_losses, test_acc, path):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
    fig.savefig(path)
    print(f"Training metrics plot is saved at {path}")

def store_classification_plot(data, path, number_of_samples=20):
    fig = plt.figure(figsize=(10, 10))
    inv_transform = get_inv_transforms()

    x_count = 5
    y_count = 1 if number_of_samples <= 5 else math.floor(number_of_samples / x_count)

    for i in range(number_of_samples):
        plt.subplot(y_count, x_count, i + 1)
        img = np.array(data[i][0].squeeze().to('cpu'))
        img = np.transpose(img, (1, 2, 0))
        img = inv_transform(image=img)['image']
        plt.imshow(img)
        plt.title(r"Correct: " + CLASS_NAMES[data[i][1].item()] + '\n' + 'Output: ' + CLASS_NAMES[data[i][2].item()])
        plt.xticks([])
        plt.yticks([])
    
    plt.tight_layout()
    plt.savefig(path)
    print(f"Sample output images are saved at {path}")