import struct

import numpy as np


def decode_idx3_ubyte(idx3_ubyte_file, num=-1):
    """decode idx3 file function

    :param num: number of files want to get
    :param idx3_ubyte_file: idx3 file path
    :return: datasets
    """
    # read binary data
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # decode header
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('magic number:%d, number of images: %d, size if images: %d*%d' %
          (magic_number, num_images, num_rows, num_cols))

    # decode datasets
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    if num != -1:
        num_images = num
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('decoded %d ' % (i + 1) + 'images')
        image = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        images[i] = np.transpose(image)
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file, num=-1):
    """decode idx1 files function

    :param num: number of files want to get
    :param idx1_ubyte_file: idx1 file path
    :return: datasets
    """
    # read binary data
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # decode header
    offset = 0
    fmt_header = '>ii'
    magic_number, num_labels = struct.unpack_from(fmt_header, bin_data, offset)
    print('magic number: %d, image number: %d' % (magic_number, num_labels))

    # decode datasets
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_labels)
    if num != -1:
        num_labels = num
    for i in range(num_labels):
        if (i + 1) % 10000 == 0:
            print('decoded %d' % (i + 1) + '')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels
