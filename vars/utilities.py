import os
import cv2
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt


def create_CSV(dir, out_dir):
    """Create a CSV file with the image names with directory path
    and their labels"""
    DIR = str(dir)+"/*/*"
    df = pd.DataFrame({'image_path': sorted(glob.glob(DIR))})
    df['class'] = df['image_path'].apply(lambda val: val.split('/')[-2])
    print(df.head())
    df.to_csv(out_dir+'Chess.csv')


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
            img = cv2.resize(img, (224, 224),
                             interpolation=cv2.INTER_CUBIC)
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
    color = ['lightblue', 'blue', 'purple', 'red', 'green', 'yellow']
    plt.figure(figsize=(10, 4))

    fig, ax = plt.subplots()
    bar_container = ax.bar(k, v, color=color)
    ax.set(ylabel='sample number', title='ChessDB', ylim=(0, 500))
    ax.bar_label(bar_container, fmt='{:,.0f}')
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
            plt.imshow(i, cmap='viridis')
            plt.tight_layout()
        lbl += 1
