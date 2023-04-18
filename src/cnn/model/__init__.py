from keras.applications import vgg16
from keras.backend import clear_session
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay
from keras.optimizers import SGD


def get_model(output_classes=15, input_shape=(32, 32, 3)):
    model = vgg16.VGG16(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=input_shape)
    for layer in model.layers:
        layer.trainable = False
    clear_session()
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu')(flat1)
    output = Dense(output_classes, activation='softmax')(class1)
    model = Model(inputs=model.inputs, outputs=output)
    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=10000,
        decay_rate=0.9)
    sgd = SGD(learning_rate=lr_schedule)  # type: ignore
    model.compile(
        optimizer=sgd,
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model
