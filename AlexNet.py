from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras import Model


def AlexNet():
    input = Input(shape=(227, 227, 3))

    x = Conv2D(96, (11, 11), strides=(4, 4), activation='relu', padding='valid')(input)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = Conv2D(256, (5, 5), strides=(1, 1), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = Conv2D(384, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(384, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1000, activation='softmax')(x)

    model = Model(inputs=[input], outputs=[output])

    return model