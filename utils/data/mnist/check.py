import os
import sys
sys.path.append(os.pardir)
import struct
import numpy as np
import matplotlib.pyplot as plt


def load_mnist(path, kind='train'):
    img_path = path + 'utils/data/mnist/' + kind + '_images'
    lab_path = path + 'utils/data/mnist/' + kind + '_labels'

    with open(lab_path, 'rb') as file_lab:
        # magic itemsがいらないなら、file_lab.seek(8)でもok
        magic, num_of_items = struct.unpack('>II', file_lab.read(8))
        labels = np.fromfile(file_lab, dtype='uint8')

    with open(img_path, 'rb') as file_img:
        magic, num_of_img, rows, cols = struct.unpack(
            '>IIII', file_img.read(16))
        images = np.fromfile(file_img, dtype='uint8').reshape(len(labels), -1)

    return images, labels, rows, cols


def main():
    images, labels, rows, cols = load_mnist('../../../')
    plt.imshow(images[4].reshape(rows, cols), cmap='gray')
    plt.xlabel(labels[4], fontsize='large')
    plt.show()
    return 0


if __name__ == '__main__':
    main()
