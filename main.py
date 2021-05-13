import decode

import matplotlib.pyplot as plt
import tensorflow as tf


from tensorflow import keras
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import LabelBinarizer


def main():
    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)

    train_images = decode.decode_idx3_ubyte("./raw_data_files/emnist-letters-train-images-idx3-ubyte")
    train_labels = decode.decode_idx1_ubyte("./raw_data_files/emnist-letters-train-labels-idx1-ubyte")
    test_images = decode.decode_idx3_ubyte("./raw_data_files/emnist-letters-test-images-idx3-ubyte")
    test_label = decode.decode_idx1_ubyte("./raw_data_files/emnist-letters-test-labels-idx1-ubyte")

    # preprocess
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    train_images = train_images / 255.0
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
    test_images = test_images / 255.0
    # one-hot encoding
    lb = LabelBinarizer()
    train_labels = lb.fit_transform(train_labels)
    test_label = lb.fit_transform(test_label)

    model = keras.Sequential([
        keras.layers.Conv2D(6, (5, 5), activation='tanh', input_shape=(28, 28, 1)),
        keras.layers.AveragePooling2D((2, 2)),
        keras.layers.Conv2D(16, (5, 5), activation='tanh'),
        keras.layers.AveragePooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(120, activation='tanh'),
        keras.layers.Dense(84, activation='tanh'),
        keras.layers.Dense(26, activation='softmax'),
    ])
    sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])
    print(model.summary())
    history = model.fit(train_images, train_labels, epochs=10, validation_split=0.1)

    test_loss, test_acc = model.evaluate(test_images, test_label, verbose=2)
    print("test, accuracy:", test_acc)

    # visualize accuracy
    plot_model(model, to_file="tanh_model.png")
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("./visualize/model_accuracy")

    # visualize losses
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("./visualize/model_loss")

    model.save("./LeNet5_model/")


if __name__ == '__main__':
    main()
