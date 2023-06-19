import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, BatchNormalization, Activation
from keras.models import Sequential
import matplotlib.pyplot as plt

NOT_AUGMENTED_TRAIN_PATH = '../cropped_dataset'
AUGMENTED_TRAIN_PATH = '../datasets/asl_new/ASL Letters.v1i.folder (1) - Copy/train/'
AUGMENTED_TEST_PATH = '../datasets/asl_new/ASL Letters.v1i.folder (1) - Copy/test/'


def get_classifier_dataset(augmented=False):
    autotune = tf.data.AUTOTUNE

    train_dataset = None
    test_dataset = None

    if not augmented:
        train_dataset_path = NOT_AUGMENTED_TRAIN_PATH
        test_dataset_path = None
    else:
        train_dataset_path = AUGMENTED_TRAIN_PATH
        test_dataset_path = AUGMENTED_TEST_PATH

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dataset_path,
        batch_size=32,
        image_size=(640, 640),
        shuffle=True,
        label_mode='categorical',
        color_mode='grayscale'
    )
    train_dataset = train_dataset.map(lambda x, y: (x / 255, y))
    train_dataset = train_dataset.prefetch(buffer_size=autotune)

    if test_dataset_path is not None:
        test_dataset = tf.keras.utils.image_dataset_from_directory(
            test_dataset_path,
            batch_size=32,
            image_size=(640, 640),
            shuffle=True,
            label_mode='categorical',
            color_mode='grayscale'
        )
        test_dataset = test_dataset.map(lambda x, y: (x / 255, y))
        test_dataset = test_dataset.cache().prefetch(buffer_size=autotune)

    return train_dataset, test_dataset


def generate_plots(history):
    acc = history.history['accuracy']
    loss = history.history['loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.title('Training accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


class Classifier:
    data = None
    classifier_model = None
    NOT_AUGMENTED_PATH = None
    AUGMENTED_PATH = None
    labels = None

    def __init__(self):
        self.NOT_AUGMENTED_MODEL_PATH = "./models/letter_classifier_model.h5"
        self.AUGMENTED_MODEL_PATH = "./models/asl_new_model_new.h5"
        self.labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Hand', 'I', 'K', 'L', 'M', 'N', 'None', 'O', 'P', 'Q',
                       'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

    def load_classifier(self):
        self.classifier_model = tf.keras.models.load_model(self.AUGMENTED_MODEL_PATH)

    def classify(self, image):
        prediction = self.classifier_model.predict(image, verbose=False)
        prediction = [(self.labels[i], int(round(prediction[0][i] * 100, 2))) for i in range(len(prediction[0]))]
        prediction.sort(key=lambda x: x[1], reverse=True)
        return prediction[0:10]

    def train_classifier_model(self, augmented=False):
        train_dataset, test_dataset = get_classifier_dataset(augmented=augmented)

        self.classifier_model = self.get_classifier_training_model(augmented=augmented)
        self.classifier_model.summary()
        self.classifier_model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
        history = self.classifier_model.fit(train_dataset,
                                            validation_data=test_dataset,
                                            epochs=8,
                                            callbacks=[
                                                tf.keras.callbacks.EarlyStopping(
                                                    monitor='loss',
                                                    patience=3,
                                                    restore_best_weights=True)
                                            ])
        if augmented:
            self.classifier_model.save(self.AUGMENTED_MODEL_PATH)
        else:
            self.classifier_model.save(self.NOT_AUGMENTED_MODEL_PATH)

        generate_plots(history)

    def train_classifier(self):
        self.train_classifier_model(augmented=False)
        self.train_classifier_model(augmented=True)

    def get_classifier_training_model(self, augmented=False):
        if augmented:
            return tf.keras.models.load_model(self.NOT_AUGMENTED_MODEL_PATH)

        return Sequential([
            Conv2D(filters=16, kernel_size=(3, 3), input_shape=(640, 640, 1)),
            BatchNormalization(),
            Activation('sigmoid'),
            MaxPool2D(2, 2, padding='same'),

            Conv2D(filters=32, kernel_size=(3, 3)),
            BatchNormalization(),
            Activation('sigmoid'),
            MaxPool2D(2, 2, padding='same'),

            Conv2D(filters=64, kernel_size=(3, 3)),
            BatchNormalization(),
            Activation('sigmoid'),
            MaxPool2D(2, 2, padding='same'),

            Conv2D(filters=128, kernel_size=(3, 3)),
            BatchNormalization(),
            Activation('sigmoid'),
            MaxPool2D(2, 2, padding='same'),

            Conv2D(filters=256, kernel_size=(3, 3)),
            BatchNormalization(),
            Activation('sigmoid'),
            MaxPool2D(2, 2, padding='same'),

            Conv2D(filters=512, kernel_size=(3, 3)),
            BatchNormalization(),
            Activation('sigmoid'),
            MaxPool2D(2, 2, padding='same'),

            Conv2D(filters=1024, kernel_size=(3, 3)),
            BatchNormalization(),
            Activation('sigmoid'),
            MaxPool2D(2, 2, padding='same'),

            Conv2D(filters=1024, kernel_size=(3, 3)),
            BatchNormalization(),
            Activation('sigmoid'),
            MaxPool2D(2, 2, padding='same'),

            Flatten(),

            Dense(units=1024),
            BatchNormalization(),
            Activation('sigmoid'),

            Dense(units=256),
            BatchNormalization(),
            Activation('sigmoid'),

            Dense(units=64),
            BatchNormalization(),
            Activation('sigmoid'),

            Dense(units=32),
            BatchNormalization(),
            Activation('sigmoid'),

            Dense(units=26),
            BatchNormalization(),
            Activation('tanh'),

            Dense(units=26),
            BatchNormalization(),
            Activation('softmax')
        ])


if __name__ == '__main__':
    classifier = Classifier()
