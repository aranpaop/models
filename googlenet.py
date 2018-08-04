from keras.layers import Input,Conv2D, BatchNormalization, Activation, MaxPooling2D, Concatenate, AveragePooling2D,\
                          Flatten, Dropout, Dense
from keras import Model


def Inception_v2(x, nb_filter):
    x_1x1 = Conv2D(nb_filter, (1, 1))(x)
    x_1x1 = BatchNormalization(axis=3)(x_1x1)
    x_1x1 = Activation('relu')(x_1x1)

    x_3x3 = Conv2D(nb_filter, (1, 1))(x)
    x_3x3 = BatchNormalization(axis=3)(x_3x3)
    x_3x3 = Activation('relu')(x_3x3)
    x_3x3 = Conv2D(nb_filter, (3, 3), padding='same')(x_3x3)
    x_3x3 = BatchNormalization(axis=3)(x_3x3)
    x_3x3 = Activation('relu')(x_3x3)

    x_3x3d = Conv2D(nb_filter, (1, 1))(x)
    x_3x3d = BatchNormalization(axis=3)(x_3x3d)
    x_3x3d = Activation('relu')(x_3x3d)
    x_3x3d = Conv2D(nb_filter, (3, 3), padding='same')(x_3x3d)
    x_3x3d = BatchNormalization(axis=3)(x_3x3d)
    x_3x3d = Activation('relu')(x_3x3d)
    x_3x3d = Conv2D(nb_filter, (3, 3), padding='same')(x_3x3d)
    x_3x3d = BatchNormalization(axis=3)(x_3x3d)
    x_3x3d = Activation('relu')(x_3x3d)

    x_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    x_pool = Conv2D(nb_filter, (1, 1))(x_pool)
    x_pool = BatchNormalization(axis=3)(x_pool)
    x_pool = Activation('relu')(x_pool)

    x = Concatenate(axis=3)([x_1x1, x_3x3, x_3x3d, x_pool])

    return x


def GoogleNet():
    input = Input(shape=(224, 224, 3))

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Conv2D(192, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = Inception_v2(x, 64)
    x = Inception_v2(x, 120)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = Inception_v2(x, 128)
    x = Inception_v2(x, 128)
    x = Inception_v2(x, 128)
    x = Inception_v2(x, 132)
    x = Inception_v2(x, 208)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = Inception_v2(x, 208)
    x = Inception_v2(x, 256)
    x = AveragePooling2D(pool_size=(7, 7), strides=(7, 7), padding='same')(x)

    x = Flatten()(x)
    x = Dropout(0.4)(x)
    x = Dense(1000, activation='relu')(x)
    output = Dense(1000, activation='softmax')(x)

    model = Model(inputs=[input], outputs=[output])

    return model