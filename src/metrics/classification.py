import tensorflow as tf
from tensorflow import keras


class SparseMacroF1(keras.metrics.Metric):
    """Compute macro F1 score for sparse integer labels.
    Converts integer class labels to one-hot encoding internally so that
    Keras' built-in F1Score metric can be used.
    """

    def __init__(self, num_classes: int, name: str = "macro_f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.f1 = keras.metrics.F1Score(average="macro", threshold=None)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=self.num_classes)
        self.f1.update_state(y_true, y_pred, sample_weight=sample_weight)

    def result(self):
        return self.f1.result()

    def reset_state(self):
        self.f1.reset_state()
