import tensorflow as tf
import matplotlib.pyplot as plt
import h5py
import keras


def load_data(dir_name, type):
    h5f = h5py.File(dir_name + type + '.h5', 'r')

    images = h5f[type + '_images'][:]
    labels = h5f[type + '_labels'][:]
    print(type + ' set', images.shape, labels.shape)

    scalar = 1 / 255.
    images = images * scalar

    h5f.close()

    return (images, labels)

def model_own():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(11, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def model_VGG_trained():
    model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    model.summary()
    
    new_output = model.output
    new_output = keras.layers.Flatten()(new_output)
    new_output = keras.layers.Dense(512, activation='relu')(new_output)
    # add new dense layer for our labels
    new_output = keras.layers.Dense(11, activation='softmax')(new_output)
    model = keras.engine.training.Model(model.inputs, new_output)
    for layer in model.layers:
        layer.trainable = True
    # for layer in model.layers[:-3]:
    #     layer.trainable = False
    model.summary()
    model.compile(keras.optimizers.Adam(lr=.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def VGG_not_trained():
    model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    model.summary()
    
    new_output = model.output
    new_output = keras.layers.Flatten()(new_output)
    new_output = keras.layers.Dense(512, activation='relu')(new_output)
    # add new dense layer for our labels
    new_output = keras.layers.Dense(11, activation='softmax')(new_output)
    model = keras.engine.training.Model(model.inputs, new_output)
    # for layer in model.layers:
    #     layer.trainable = True
    for layer in model.layers[:-3]:
        layer.trainable = False
    model.summary()
    model.compile(keras.optimizers.Adam(lr=.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def model_train(model, X_train, y_train, X_valid, y_valid, X_test, y_test, name):
    batch_size = 128
    num_epoch = 6

    model_log = model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=num_epoch,
              verbose=1,
              validation_data=(X_valid, y_valid))

    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    model.save_weights("%s.h5" % name)
    print("Saved model to disk")
    
    return model_log

def output(model_log, name):
    plt.subplot(2,1,1)
    plt.plot(model_log.history['acc'])
    plt.plot(model_log.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.subplot(2,1,2)
    plt.plot(model_log.history['loss'])
    plt.plot(model_log.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.tight_layout()
    plt.savefig("%s.png" % name)

if __name__ == '__main__':
    dir_name = 'data/cropped/'
    X_train, y_train = load_data(dir_name, "train")
    X_valid, y_valid = load_data(dir_name, "test")
    X_test, y_test = load_data(dir_name, "valid")
    
    model = model_own()
    model_log = model_train(model, X_train, y_train, X_valid, y_valid, X_test, y_test, 'model_own')
    output(model_log, 'model_own')
    
    model = model_VGG_trained()
    model_log = model_train(model, X_train, y_train, X_valid, y_valid, X_test, y_test, 'model_VGG_trained')
    output(model_log,'model_VGG_trained')
    
    model=VGG_not_trained()
    model_log=model_train(model, X_train, y_train, X_valid, y_valid, X_test, y_test, 'VGG_not_trained')
    output(model_log,'VGG_not_trained')