import tensorflow as tf
import keras.backend as K


# def tversky(alpha=0.5, beta=0.5, axis=(0, 1, 2), smooth=1e-5):
#     def loss(y_true, y_pred):
#         ones = tf.ones_like(y_true, dtype=tf.float32)
#         p0 = y_pred
#         p1 = ones - y_pred
#         g0 = y_true
#         g1 = ones - y_true
#
#         tp = tf.reduce_sum(p0 * g0, axis=axis)
#         fp = tf.reduce_sum(p0 * g1, axis=axis)
#         fn = tf.reduce_sum(p1 * g0, axis=axis)
#
#         deno = tp + alpha * fp + beta * fn
#
#         t_loss = 21 - tf.reduce_sum(tp / (deno + smooth))
#
#         return t_loss
#
#     return loss


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')

    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss
