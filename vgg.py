from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from keras import Input, Model


def get_vgg16():
    input = Input(shape=(224, 224, 3))

    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(c5)
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(c5)
    p5 = MaxPooling2D((2, 2))(c5)

    f6 = Flatten()(p5)
    d6 = Dense(4096, activation='relu')(f6)
    dr6 = Dropout(0.5)(d6)

    d7 = Dense(4096, activation='relu')(dr6)
    dr7 = Dropout(0.5)(d7)

    d8 = Dense(1000, activation='softmax')(dr7)
    output = Dropout(0.5)(d8)

    model = Model(inputs=[input], outputs=[output])

    return model


def get_vgg19():
    input = Input(shape=(224, 224, 3))

    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(c5)
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(c5)
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(c5)
    p5 = MaxPooling2D((2, 2))(c5)

    f6 = Flatten()(p5)
    d6 = Dense(4096, activation='relu')(f6)
    dr6 = Dropout(0.5)(d6)

    d7 = Dense(4096, activation='relu')(dr6)
    dr7 = Dropout(0.5)(d7)

    d8 = Dense(1000, activation='softmax')(dr7)
    output = Dropout(0.5)(d8)

    model = Model(inputs=[input], outputs=[output])

    return model