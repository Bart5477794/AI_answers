#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 14:47:03 2022

AIRR gate detection

@author: guido
"""

# https://towardsdatascience.com/beginners-guide-to-loading-image-data-with-pytorch-289c60b7afec

# idea for lab session: load a pretrained UNet and *refine* it for the drone racing task
# https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torch
import os
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Lambda
import torchvision.transforms as Transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from loss_functions import mixed_loss, dice_loss


def load_AIRR_images(image_dir='./MAVLAB_TUDelft_gate_dataset/gate_dataset/', postfix='.png'):
    """ 
    Load the images and masks
    """

    images = []
    masks = []
    files = os.listdir(image_dir)
    files.sort()
    for file in files:

        if (file.endswith(postfix)):
            im = Image.open(image_dir + file)
            width, height = im.size
            size = 256  # We need this size for the preloaded UNet (it requires powers of 2)
            if (width != size or height != size):
                im = im.resize((size, size))
            im = np.array(im)

            if (file.startswith('img')):
                images.append(im)
            elif (file.startswith('mask')):
                masks.append(im)

    return [np.asarray(masks), np.asarray(images)]


# Utility functions for transforming images and masks from/to PIL images and tensors:
transform_tensor = Transforms.ToTensor()
transform_PIL = Transforms.ToPILImage()


class AIRR_Dataset(Dataset):
    """ 
    Dataset class for the AIRR dataset
    """

    def __init__(self):

        print('Load images')
        [masks, images] = load_AIRR_images()
        self.X = images
        self.t = masks

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        p = np.random.randint(0, 2, 1)[0]

        # Images:
        im = self.X[idx]
        im = transform_tensor(im)  # at least make it a tensor
        if p == 1:
            im = torch.flip(im, [1])

        # Masks:
        t = self.t[idx]
        t = torch.tensor(t)
        t = torch.reshape(t, (1, t.shape[0], t.shape[1]))
        t = t.float() / 255.0
        if p == 1:
            t = torch.flip(t, [1])

        sample = im, t
        return sample


def train_loop(dataloader, model, loss_fn, optimizer):
    [X, y] = next(iter(dataloader))
    size = len(y)

    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    print(f"Train Error: \n Avg loss: {loss:>8f} \n")


def validation_loop(dataloader, model, loss_fn):
    validation_loss = 0.

    [X, y] = next(iter(dataloader))
    size = len(y)

    with torch.no_grad():
        # Compute prediction and loss
        pred = model(X)
        validation_loss += loss_fn(pred, y).item()

    print(f"Validation Error: \n Avg loss: {validation_loss:>8f} \n")


def train_U_Net(learning_rate=1e-2, batch_size=10, epochs=50, gen=None):

    dataset = AIRR_Dataset()

    # Make a training and validation set:
    training_ratio = 0.8
    n_samples = dataset.__len__()
    indices = list(range(n_samples))
    split = int(np.floor(training_ratio * n_samples))
    train_inds = indices[:split]
    val_inds = indices[split:]
    train_sampler = SubsetRandomSampler(train_inds, generator=gen)
    valid_sampler = SubsetRandomSampler(val_inds, generator=gen)

    """
    TODO:
    Part 2: 
        - Switch the loss function to {dice_loss} and {mixed_loss} to 
            test your implementation
    """

    # Learning settings:
    loss_fn = mixed_loss  # binary cross-entropy loss

    # Load the UNet:
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                           in_channels=3, out_channels=1, init_features=32, pretrained=True)
    # Set the optimizer:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        validation_loop(val_dataloader, model, loss_fn)
    print("Done!")

    torch.save(model.state_dict(), 'model_weights_AIRR.pth')


if __name__ == '__main__':

    seed = 0
    rn_gen = torch.Generator()
    rn_gen.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    train = True

    if (train):
        train_U_Net(gen=rn_gen)
    else:
        dataset = AIRR_Dataset()

        # Load the UNet:
        model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                               in_channels=3, out_channels=1, init_features=32, pretrained=True)
        model.load_state_dict(torch.load('model_weights_AIRR.pth'))
        model.eval()

        # Loop over the images:
        i = 0
        while (True):
            inp = input('Press enter to show next image, or type "e" or "exit" to quit: ')

            # Exit:
            if (inp == 'exit' or inp == 'e'):
                break

            [im, mask] = dataset.__getitem__(i)
            RGB = transform_PIL(im)
            mask = transform_PIL(mask)

            plt.figure()
            plt.imshow(RGB)
            plt.savefig('rgb.png')
            plt.close()

            plt.figure()
            plt.imshow(mask)
            plt.colorbar()
            plt.savefig('mask.png')
            plt.close()

            x = torch.reshape(im, (1, im.shape[0], im.shape[1], im.shape[2]))
            out = model(x)

            out = torch.reshape(out, (1, out.shape[2], out.shape[3]))
            mask_pred = transform_PIL(out)
            plt.figure()
            plt.imshow(mask_pred)
            plt.colorbar()
            plt.savefig('mask_pred.png')
            plt.close()

            i += 1
