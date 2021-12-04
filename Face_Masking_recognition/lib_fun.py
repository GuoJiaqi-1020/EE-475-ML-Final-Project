import numpy as np
import matplotlib.pyplot as plt
import cv2
import sklearn.svm as svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from skimage import feature
import os

def RGBtoGray(data):
    new_data = np.zeros((1, data.shape[1], data.shape[2]))
    for im in data:
        new_data = np.append(new_data, cv2.cvtColor(im.astype('float32'),cv2.COLOR_RGB2GRAY).reshape(1,data.shape[1], data.shape[2]), axis=0)
    return new_data[1:,:,:]

def RGBtoHOG(data):
    new_data = np.zeros((1, data.shape[1], data.shape[2]))
    for im in data:
        fd, hog_image = feature.hog(im, orientations=8, pixels_per_cell=(2, 2),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)
        new_data = np.append(new_data, hog_image.reshape((1, data.shape[1], data.shape[2])), axis=0)
    return new_data[1:,:,:]

def GRAYtoEDGE(data, sigma):
    new_data = np.zeros((1, data.shape[1], data.shape[2]))
    for im in data:
        edges = feature.canny(im, sigma=sigma)
        new_data = np.append(new_data, edges.reshape((1, data.shape[1], data.shape[2])), axis=0)
    return new_data[1:,:,:]

def Flatten(images):
    images = images.reshape(images.shape[0], -1)
    return images

def load(data_path):
    for i, set in enumerate(['Correct', 'Incorrect', 'NoMask']):
        if i == 0:
            data = np.load(data_path+set+'.npy')
            labels = np.array([i]*len(data))
            i += 1
        else:
            data_ = np.load(data_path+set+'.npy')
            data = np.append(data,data_, axis=0)
            labels = np.append(labels, np.array([i]*len(data_)), axis=0)
    print('X shape: {}, Y shape: {}'.format(data.shape, np.array(labels).shape))
    return data, labels

def VisualizeRGB(data20, data50, data100):
    f, axes = plt.subplots(1,9, figsize=(18,2))
    axes[0].imshow(data20[0,:,:,:].astype('uint8'))
    axes[1].imshow(data20[2000,:,:,:].astype('uint8'))
    axes[2].imshow(data20[-1,:,:,:].astype('uint8'))
    axes[3].imshow(data50[0,:,:,:].astype('uint8'))
    axes[4].imshow(data50[2000,:,:,:].astype('uint8'))
    axes[5].imshow(data50[-1,:,:,:].astype('uint8'))
    axes[6].imshow(data100[0,:,:,:].astype('uint8'))
    axes[7].imshow(data100[2000,:,:,:].astype('uint8'))
    axes[8].imshow(data100[-1,:,:,:].astype('uint8'))
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

def VisualizeGray(data20, data50, data100):
    f, axes = plt.subplots(1,9, figsize=(18,2))
    axes[0].imshow(data20[0,:,:].astype('uint8'),cmap ='gray')
    axes[1].imshow(data20[2000,:,:].astype('uint8'),cmap ='gray')
    axes[2].imshow(data20[-1,:,:].astype('uint8'),cmap ='gray')
    axes[3].imshow(data50[0,:,:].astype('uint8'),cmap ='gray')
    axes[4].imshow(data50[2000,:,:].astype('uint8'),cmap ='gray')
    axes[5].imshow(data50[-1,:,:].astype('uint8'),cmap ='gray')
    axes[6].imshow(data100[0,:,:].astype('uint8'),cmap ='gray')
    axes[7].imshow(data100[2000,:,:].astype('uint8'),cmap ='gray')
    axes[8].imshow(data100[-1,:,:].astype('uint8'),cmap ='gray')
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])


def plot_confusion_matrix(cm_collection, labels=["Correct Masking", "Incorrect Masking", "No Masking"], precision="%0.f", normalize=True):
    tick_marks = np.array(range(len(labels))) + 0.5
    title = ["Image Size = 20", "Image Size = 50", "Image Size = 100"]
    plt.figure(figsize=(16, 8), dpi=120)
    for i in range(len(cm_collection)):
        cm = cm_collection[i]
        plt.subplot(1, len(cm_collection), i + 1)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title[i] = "Normalized " + title[i]
            precision = "%0.2f"
        ind_array = np.arange(len(labels))
        x, y = np.meshgrid(ind_array, ind_array)
        for x_val, y_val in zip(x.flatten(), y.flatten()):
            c = cm[y_val][x_val]
            if c > 0.0:
                plt.text(x_val, y_val, precision % (c,), color='k', fontsize=11, va='center', ha='center')
        plt.gca().set_xticks(tick_marks, minor=True)
        plt.gca().set_yticks(tick_marks, minor=True)
        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')
        plt.grid(True, which='minor', linestyle='-')
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        font = {'size': 9}
        plt.title(title[i], font)
        xlocations = np.array(range(len(labels)))
        plt.xticks(xlocations, labels, rotation=45)
        if i == 0:
            plt.yticks(xlocations, labels)
        else:
            plt.yticks([])
    plt.show()


# def plot_confusion(confusion):
#     labels = ["Correct Masking", "Incorrect Masking", "No Masking"]
#     title = "Confusion matrix: Raw Dataset"
#     tick_marks = np.array(range(len(labels))) + 0.5
#     cm = confusion
#     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#     title = "Normalized " + title
#     precision = "%0.2f"
#     plt.figure(figsize=(6, 4), dpi=120)
#     ind_array = np.arange(len(labels))
#     x, y = np.meshgrid(ind_array, ind_array)
#     for x_val, y_val in zip(x.flatten(), y.flatten()):
#         c = cm[y_val][x_val]
#         if c > 0.0:
#             plt.text(x_val, y_val, precision % (c,), color='k', fontsize=8, va='center', ha='center')
#     plt.gca().set_xticks(tick_marks, minor=True)
#     plt.gca().set_yticks(tick_marks, minor=True)
#     plt.gca().xaxis.set_ticks_position('none')
#     plt.gca().yaxis.set_ticks_position('none')
#     plt.grid(True, which='minor', linestyle='-')
#     plt.gcf().subplots_adjust(bottom=0.15)
#     plt.imshow(cm, interpolation='nearest', cmap='Blues')
#     font = {'size': 5}
#     plt.title(title, font)
#     plt.colorbar()
#     xlocations = np.array(range(len(labels)))
#     plt.xticks(xlocations, labels, rotation=0, fontsize=font['size'])
#     plt.yticks(xlocations, labels, fontsize=font['size'])
#     plt.ylabel('True label', font)
#     plt.xlabel('Predicted label', font)
#     plt.show()
