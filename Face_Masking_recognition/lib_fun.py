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

def loads(path1, path2, path3):
    data20, labels20 = load(path1)
    data50, labels50 = load(path2)
    data100, labels100 = load(path3)
    VisualizeRGB(data20, data50, data100)
    return data20, labels20, data50, labels50, data100, labels100

def RGBtoGrays(data1, data2, data3):
    data20_gray = RGBtoGray(data1)
    data50_gray = RGBtoGray(data2)
    data100_gray = RGBtoGray(data3)
    VisualizeGray(data20_gray, data50_gray, data100_gray)
    return data20_gray, data50_gray, data100_gray

def RGBtoHOGs(data1, data2, data3):
    data20_hog = RGBtoHOG(data1)
    data50_hog = RGBtoHOG(data2)
    data100_hog = RGBtoHOG(data3)
    VisualizeGray(data20_hog, data50_hog, data100_hog)
    return data20_hog, data50_hog, data100_hog

def GraytoCanny(data1, data2, data3, sigma):
    data20_edge = GRAYtoEDGE(data1, sigma=sigma[0])
    data50_edge = GRAYtoEDGE(data2, sigma=sigma[1])
    data100_edge = GRAYtoEDGE(data3, sigma=sigma[2])
    VisualizeGray(data20_edge, data50_edge, data100_edge)
    return data20_edge, data50_edge, data100_edge

# Training process using different model. Split the data into training : test = 4 : 1.
# Flatten the image as the input data of model
def train_model(model, data, labels):
    x = Flatten(data) 
    x_train, x_test , y_train, y_test = train_test_split(x, labels, test_size = 0.2, random_state=1)
    clf = make_pipeline( StandardScaler(), model)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    print('accuracy: ', metrics.accuracy_score(y_test, pred)) 
    confusion = metrics.confusion_matrix(y_test, pred)
    return confusion

def train_models(model_fun, data1, data2, data3, label1, label2, label3):
    con1 = train_model(model_fun, data1, label1)
    con2 = train_model(model_fun, data2, label2)
    con3 = train_model(model_fun, data3, label3)
    return con1, con2, con3


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


def plot_confusion_matrix(cm_collection, labels=["Correct", "Incorrect", "No"], precision="%0.f", normalize=True):
    tick_marks = np.array(range(len(labels))) + 0.5
    title = ["20*20", "50*50", "100*100"]
    plt.figure(figsize=(4, 2), dpi=120)
    for i in range(len(cm_collection)):
        cm = cm_collection[i]
        plt.subplot(1, len(cm_collection), i + 1)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title[i] = title[i]
            precision = "%0.2f"
        ind_array = np.arange(len(labels))
        x, y = np.meshgrid(ind_array, ind_array)
        for x_val, y_val in zip(x.flatten(), y.flatten()):
            c = cm[y_val][x_val]
            if c > 0.0:
                plt.text(x_val, y_val, precision % (c,), color='k', fontsize=8, va='center', ha='center')
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
        plt.xticks(xlocations, labels, rotation=45, fontsize=6)
        if i == 0:
            plt.yticks(xlocations, labels, fontsize=6)
        else:
            plt.yticks([])
    plt.show()
