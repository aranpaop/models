from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras import Model


def LeNet():
    input = Input(shape=(28, 28, 1))

    x = Conv2D(32, (5, 5), activation='relu', padding='valid')(input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (5, 5), activation='relu', padding='valid')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    output = Dense(10, activation='softmax')(x)

    model = Model(inputs=[input], outputs=[output])

    return model