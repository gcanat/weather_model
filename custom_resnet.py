from tensorflow import keras
from tensorflow.kears import layers


class ResBlock(keras.Model):
    def __init__(self, filters, downsample):
        super().__init__()
        if downsample:
            self.conv1 = layers.Conv2D(filters, 3, 2, padding='same')
            self.shortcut = keras.Sequential([
                layers.Conv2D(filters, 1, 2),
                layers.BatchNormalization()
            ])
        else:
            self.conv1 = layers.Conv2D(filters, 3, 1, padding='same')
            self.shortcut = keras.Sequential()
 
        self.conv2 = layers.Conv2D(filters, 3, 1, padding='same')
    def call(self, input):
        shortcut = self.shortcut(input)

        input = self.conv1(input)
        input = layers.BatchNormalization()(input)
        input = layers.ReLU()(input)

        input = self.conv2(input)
        input = layers.BatchNormalization()(input)
        input = layers.ReLU()(input)

        input = input + shortcut
        return layers.ReLU()(input)


class ResNet18(keras.Model):
    def __init__(self, outputs=1000):
        super().__init__()
        self.layer0 = keras.Sequential([
            layers.Conv2D(64, 7, 2, padding='same'),
            layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ], name='layer0')

        self.layer1 = keras.Sequential([
            ResBlock(64, downsample=False),
            ResBlock(64, downsample=False)
        ], name='layer1')

        self.layer2 = keras.Sequential([
            ResBlock(128, downsample=True),
            ResBlock(128, downsample=False)
        ], name='layer2')

        self.layer3 = keras.Sequential([
            ResBlock(256, downsample=True),
            ResBlock(256, downsample=False)
        ], name='layer3')

        self.layer4 = keras.Sequential([
            ResBlock(512, downsample=True),
            ResBlock(512, downsample=False)
        ], name='layer4')

        self.gap = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(outputs, activation='softmax')
    def call(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.gap(input)
        input = self.fc(input)

        return input
