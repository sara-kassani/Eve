import os        
import json
import models
import numpy as np
# from keras.utils import np_utils
# from keras.datasets import cifar10, cifar100, mnist
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from Eve import Eve


def train(model_name, **kwargs):
    """
    Train model

    args: model_name (str, keras model name)
          **kwargs (dict) keyword arguments that specify the model hyperparameters
    """

    # Roll out the parameters
    batch_size = kwargs["batch_size"]
    nb_epoch = kwargs["nb_epoch"]
    # dataset = kwargs["dataset"]
    optimizer = kwargs["optimizer"]
    experiment_name = kwargs["experiment_name"]

    # Compile model.
    if optimizer == "SGD":
        opt = SGD(lr=1E-2, decay=1E-4, momentum=0.9, nesterov=True)
    if optimizer == "Adam":
        opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1E-4)
    if optimizer == "Eve":
        opt = Eve(lr=1E-4, decay=1E-4, beta_1=0.9, beta_2=0.999, beta_3=0.999, small_k=0.1, big_K=10, epsilon=1e-08)

    # if dataset == "endovis":
        # (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    dataset = "endovis"
    # if dataset == "cifar100":
    #     (X_train, y_train), (X_test, y_test) = cifar100.load_data()
    # if dataset == "mnist":
    #     (X_train, y_train), (X_test, y_test) = mnist.load_data()
    #     X_train = X_train.reshape((X_train.shape[0], 1, 28, 28))
    #     X_test = X_test.reshape((X_test.shape[0], 1, 28, 28))
    #
    # X_train = X_train.astype('float32')
    # X_test = X_test.astype('float32')
    # X_train /= 255.
    # X_test /= 255.
    #
    # img_dim = X_train.shape[-3:]
    # nb_classes = len(np.unique(y_train))
    #
    # # convert class vectors to binary class matrices
    # Y_train = np_utils.to_categorical(y_train, nb_classes)
    # Y_test = np_utils.to_categorical(y_test, nb_classes)


    train_dir = 'chest_xray/train'
    valid_dir = 'chest_xray/val'
    test_dir = 'chest_xray/test'
    img_width, img_height = 75, 75
    batch_size = 16
    num_epochs = 2
    filter_size = (3, 3)
    pool_size = (2, 2)
    drop_out_dense = 0.5
    drop_out_conv = 0.25
    padding = 'same'
    img_dim = (img_width, img_height, 1)
    nb_classes= 2


    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale')


    validation_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True)
    validation_generator = validation_datagen.flow_from_directory(
        valid_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale')

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale')






    # Compile model
    model = models.load(model_name, img_dim, nb_classes)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    model.summary()
    for e in range(nb_epoch):

        loss = model.fit_generator(train_generator,
                          steps_per_epoch= 400,
                          validation_data = validation_generator,
                          validation_steps=100,
                          epochs=1)


        train_losses.append(loss.history["loss"])
        val_losses.append(loss.history["val_loss"])
        train_accs.append(loss.history["acc"])
        val_accs.append(loss.history["val_acc"])

        # Save experimental log
        d_log = {}
        d_log["experiment_name"] = experiment_name
        d_log["img_dim"] = img_dim
        d_log["batch_size"] = batch_size
        d_log["nb_epoch"] = nb_epoch
        d_log["train_losses"] = train_losses
        d_log["val_losses"] = val_losses
        d_log["train_accs"] = train_accs
        d_log["val_accs"] = val_accs
        d_log["optimizer"] = opt.get_config()
        # Add model architecture
        json_string = json.loads(model.to_json())
        for key in json_string.keys():
            d_log[key] = json_string[key]
        json_file = os.path.join("log", '%s_%s_%s.json' % (dataset, model.name, experiment_name))
        with open(json_file, 'w') as fp:
            json.dump(d_log, fp, indent=4, sort_keys=True)
