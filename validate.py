import decode

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras


def main():
    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)

    classes_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                     's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    path = "./test/normal/q.png"  # real test image path

    model = keras.models.load_model("LeNet5_model")
    # print(model.summary())

    # test_data_validate(model, classes_names)
    real_write_test(model, classes_names, path)


def test_data_validate(model, classes_names):
    """checkout test images recognition result

    choose ten of them, show images in plt and result in console

    :param model: trained LeNet5 model
    :param classes_names: readable classes names
    :return: none
    """
    test_images = decode.decode_idx3_ubyte("raw_data_files/emnist-letters-train-images-idx3-ubyte", 50)
    test_images = test_images / 255.0
    for i in range(10):
        test_image = test_images[i]
        plt.figure()
        plt.imshow(test_image)
        plt.colorbar()
        test_image = test_image.reshape(28, 28, 1)
        test_image = np.expand_dims(test_image, 0)
        predictions = model.predict(test_image)
        plt.xlabel("This is " + classes_names[np.argmax(predictions[0])])
        plt.xticks([])
        plt.yticks([])
        plt.show(block=True)


def real_write_test(model, classes_names, path):
    """test the real write characters by hand

    :param model: trained LeNet5 model
    :param classes_names: readable classes names
    :param path: test image path
    :return: none
    """
    test_img = cv2.imread(path)

    # preprocess
    test_img = cv2.resize(test_img, (28, 28))
    plt.figure()
    plt.imshow(test_img)
    plt.colorbar()
    plt.yticks([])
    plt.xticks([])
    plt.show()
    test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)
    plt.figure()
    plt.imshow(test_img)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.show()
    test_img = np.array(test_img)

    test_img = 255 - test_img
    test_img = (test_img - np.min(test_img)) / (np.max(test_img) - np.min(test_img))

    # simple filter, enhance signal
    for i in range(28):
        for j in range(28):
            if test_img[i][j] < 0.2:
                test_img[i][j] = 0
            elif test_img[i][j] >= 0.8:
                test_img[i][j] = 1

    plt.figure()
    plt.imshow(test_img)
    test_img = test_img.reshape(28, 28, 1)
    test_img = np.expand_dims(test_img, 0)
    predictions = model.predict(test_img)
    plt.yticks([])
    plt.xticks([])
    plt.colorbar()
    plt.xlabel("This is " + classes_names[np.argmax(predictions[0])])
    plt.show()


if __name__ == '__main__':
    main()
