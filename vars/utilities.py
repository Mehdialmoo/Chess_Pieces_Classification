import os
import cv2
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt


def create_CSV(dir):
    df = pd.DataFrame({'image_path': sorted(glob.glob(dir))})
    df['class'] = df['image_path'].apply(lambda x: x.split('/')[-2])
    df.to_csv(dir+'/out.csv')


def pre_process(dir_list):
    """"DocString"""
    # Loop through each sub-directory (class) and get a list of all image paths
    for sub_dir in dir_list:
        image_paths = glob.glob(os.path.join(sub_dir, '*jpg'))
        # Loop through each image path and resize it
        i = 0
        for image_path in image_paths:
            i += 1
            img = cv2.imread(image_path)
            """img = cv2.resize
            (img, (224, 224), interpolation = cv2.INTER_CUBIC)"""
            img = img.astype(np.float32)
            cv2.imwrite(image_path, img)
            os.rename(
                image_path, os.path.join(sub_dir, ''.join([str(i), '.jpg'])))


def plot_bar(dir, labels):
    """this function plots a barchart that ables the user to
       visulise the size of samples in each class"""

    image_counts = [
        len(os.listdir(os.path.join(dir, class_name)))
        for class_name in labels]

    d_dict = dict((labels[i], image_counts[i]) for i in range(len(labels)))

    k = d_dict.keys()
    v = d_dict.values()
    plt.figure(figsize=(10, 4))
    bars = plt.bar(k, v)
    bars[0].set_color('green')
    bars[1].set_color('black')
    bars[2].set_color('blue')
    bars[3].set_color('yellow')
    bars[4].set_color('purple')
    bars[5].set_color('orange')
    plt.show()


def plot_img(dir_list, labels, image_no=8):
    """this  function visulises samples from each class """
    lbl = 0
    for path in dir_list:
        tt = os.listdir(path)
        items = tt[:image_no]
        plt.figure(figsize=(15, 15))
        for indx, img in enumerate(items):
            plt.subplot(8, 8, indx+1)
            i_d = os.path.join(path, img)
            i = plt.imread(i_d, 0)
            plt.title(labels[lbl])
            plt.imshow(i)
            plt.imshow(i, cmap='viridis')
            lbl += 1
            plt.tight_layout()