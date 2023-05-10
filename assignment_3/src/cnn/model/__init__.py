from keras.applications import vgg16
from keras.applications import ResNetRS50
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
# from keras.optimizers.optimizer_experimental.adamw import AdamW


def get_model(output_classes=15, input_shape=(32, 32, 3)):
    model = vgg16.VGG16(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=input_shape)
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(240, activation='relu')(flat1)
    output = Dense(output_classes, activation='softmax')(class1)
    model = Model(inputs=model.inputs, outputs=output)
    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.07,
        decay_steps=5000,
        decay_rate=0.87)
    optim = SGD(learning_rate=lr_schedule)  # type: ignore
    # optim = AdamW(
    #     learning_rate=0.01,
    #     weight_decay=0.01,
    # )
    model.compile(
        optimizer=optim,
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model


def get_model_resnet(output_classes=15, input_shape=(200, 200, 3)):
    model = ResNetRS50(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=input_shape)
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(240, activation='relu')(flat1)
    output = Dense(output_classes, activation='softmax')(class1)
    model = Model(inputs=model.inputs, outputs=output)
    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.07,
        decay_steps=5000,
        decay_rate=0.87)
    optim = SGD(learning_rate=lr_schedule)  # type: ignore
    # optim = AdamW(
    #     learning_rate=0.01,
    #     weight_decay=0.01,
    # )
    model.compile(
        optimizer=optim,
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model


def save_best_callback(model_save_path):
    return ModelCheckpoint(
        filepath=model_save_path,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False,
        verbose=1)
