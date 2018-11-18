import tensorflow as tf
import keras.backend as K


# Define IoU metric
# IOU = true_positive / (true_positive + false_positive + false_negative).
def mean_iou(num_class:int):
    def mean_iou(y_true, y_pred):
        y = tf.argmax(y_true, -1)
        y_hat = tf.argmax(y_pred, -1)
        score, up_opt = tf.metrics.mean_iou(y, y_hat, num_class)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        return score
    return mean_iou
