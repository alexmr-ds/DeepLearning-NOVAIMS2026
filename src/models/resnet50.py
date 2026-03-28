from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D


def build_model(num_classes: int = 23, trainable_layers: int = 30) -> keras.Model:
    """
    Build a transfer learning model based on ResNet50.

    This model uses a ResNet50 backbone pre-trained on ImageNet, applies
    data augmentation during training, freezes all layers except the last
    `trainable_layers`, and adds a custom classification head.

    Parameters
    ----------
    num_classes : int, default=23
        Number of output classes.
    trainable_layers : int, default=30
        Number of top layers in the ResNet50 backbone to unfreeze for fine-tuning.

    Returns
    -------
    keras.Model
        A Keras model ready for compilation and training.
    """

    # --- Data augmentation --- #
    # inside the model
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom((-0.15, 0.10)),
            layers.RandomTranslation(0.05, 0.05),
            layers.RandomBrightness(0.08, value_range=(0, 255)),
            layers.RandomContrast(0.12, value_range=(0, 255)),
            layers.RandomSaturation((0.45, 0.60), value_range=(0, 255)),
            layers.RandomSharpness((0.45, 0.60), value_range=(0, 255)),
        ],
        name="data_augmentation",
    )

    # --- Base model --- #
    base_model = ResNet50(weights="imagenet", include_top=False)

    # Enable fine-tuning: freeze all but the last `trainable_layers`
    base_model.trainable = True
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False

    # --- Model pipeline --- #
    inputs = keras.Input(shape=(224, 224, 3))

    x = data_augmentation(inputs)
    x = preprocess_input(x)  # Apply ResNet50-specific preprocessing
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs)
