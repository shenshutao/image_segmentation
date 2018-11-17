import keras.backend as K
import tensorflow as tf
import random
import numpy as np
from keras.utils.np_utils import to_categorical


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


def binary_crossentropy(batch_y_true, batch_y_pred):
    """
    :param batch_y_true: 维度为 [batch_size]
    :param batch_y_pred: 维度为 [batch_size]
    """
    # 下式中，给batch_y_pred加上了一个epsilon 1e-10, log(0)值为-inf，tf里面 0 * -inf = nan。

    # 先求batch中每个样本的cross entropy值
    batch_y_pred = tf.clip_by_value(batch_y_pred, 1e-10, 1-1e-10)  # 防止log(0)为-inf，tf里面 0 * -inf = nan。
    batch_loss = batch_y_true * tf.log(batch_y_pred) + (1 - batch_y_pred) * tf.log(1 - batch_y_pred)
    # 再对这个batch中的loss求平均值
    loss = - tf.reduce_mean(batch_loss)

    return loss


def categorical_crossentropy(batch_y_true, batch_y_pred):
    """
    :param batch_y_true: 维度为 [batch_size, class_no]
    :param batch_y_pred: 维度为 [batch_size, class_no]
    """
    # 先求batch中每个样本的cross entropy值
    batch_y_pred = tf.clip_by_value(batch_y_pred, 1e-10, 1)  # 防止log(0)为-inf，tf里面 0 * -inf = nan。
    batch_loss = tf.reduce_sum(batch_y_true * tf.log(batch_y_pred), 1)
    # 再对这个batch中的loss求平均值
    loss = - tf.reduce_mean(batch_loss)
    return loss


def weighted_categorical_crossentropy(weights):
    """
    在categorical_crossentropy的基础上，为每个类别返回的loss值加上权重
    :param weights: 维度为 [class_no]

    用法：
    loss = weighted_categorical_crossentropy([0.5, 1, 2])  # 类别的loss权重分别为 0.5, 1, 2
    model.compile(loss=loss,optimizer='adam')
    """
    weights = tf.constant(weights, dtype=tf.float32)

    def loss(batch_y_true, batch_y_pred):
        """
        :param batch_y_true: 维度为 [batch_size, class_no]
        :param batch_y_pred: 维度为 [batch_size, class_no]
        """
        # 先求batch中每个样本的cross entropy值
        batch_y_pred = tf.clip_by_value(batch_y_pred, 1e-10, 1)
        batch_loss = tf.reduce_sum(batch_y_true * tf.log(batch_y_pred) * weights, 1)
        # 再对这个batch中的loss求平均值
        loss = - tf.reduce_mean(batch_loss)
        return loss

    return loss


def segmentation_weighted_categorical_crossentropy(weights):
    """
    在categorical_crossentropy的基础上，为每个类别返回的loss值加上权重
    :param weights: 维度为 [class_no]

    用法：
    loss = weighted_categorical_crossentropy([0.5, 1, 2])  # 类别的loss权重分别为 0.5, 1, 2
    model.compile(loss=loss,optimizer='adam')
    """
    weights = tf.constant(weights, dtype=tf.float32)

    def loss(batch_hw_y_true, batch_hw_y_pred):
        """
        :param batch_y_true: 维度为 [batch_size, image_H, image_W, class_no]
        :param batch_y_pred: 维度为 [batch_size, image_H, image_W, class_no]
        """
        print(K.int_shape(batch_hw_y_true))
        print(K.int_shape(batch_hw_y_pred))
        print(K.dtype(batch_hw_y_true))
        print(K.dtype(batch_hw_y_pred))

        # clip pred value, 防止log(0)
        batch_hw_y_pred = tf.clip_by_value(batch_hw_y_pred, 1e-10, 1)
        # 先求batch中每个样本的cross entropy值, weights在做乘法的时候会自动broadcast
        batch_loss = tf.reduce_sum(batch_hw_y_true * tf.log(batch_hw_y_pred) * weights, 3)
        # 再对这个batch中的loss求平均值
        loss = - tf.reduce_mean(batch_loss)
        return loss

    return loss


if __name__ == "__main__":
    with tf.Session() as sess:
        batch_bin_y_true = tf.constant([1., 0., 0.])
        batch_bin_y_pred = tf.constant([0, 0.1, 0.5])
        loss1 = binary_crossentropy(batch_bin_y_true, batch_bin_y_pred)
        print(sess.run(loss1))

        batch_cat_y_true = tf.constant([[1., 0., 0.], [0., 1., 0.]])
        batch_cat_y_pred = tf.constant([[0., 0., 1.], [0.4, 0.1, 0.5]])

        loss2 = categorical_crossentropy(batch_cat_y_true, batch_cat_y_pred)
        print(sess.run(loss2))

        wcce = weighted_categorical_crossentropy([0.5, 1, 2])
        loss3 = wcce(batch_cat_y_true, batch_cat_y_pred)
        print(sess.run(loss3))

        # Batch:2 H:2 W:2 Class_no:3
        batch_hw_cat_y_true = tf.constant([[[[1., 0., 0.], [0., 1., 1.]], [[1., 0., 0.], [0., 1., 0.]]], [[[1., 0., 0.], [0., 1., 1.]], [[1., 0., 0.], [0., 1., 0.]]]])
        batch_hw_cat_y_pred = tf.constant([[[[0., 0., .2], [1., 1., .8]], [[.4, .1, .3], [.5, .7, .1]]], [[[1., 0., 0.], [0., 1., 1.]], [[1., 0., 0.], [0., 1., 0.]]]])

        logits = tf.random_uniform(shape=[5, 224, 224, 3], minval=0, maxval=1, dtype=tf.float32)

        # segmentation的标注一般是一个mask图片，每个像素一个类别，one-hot encoding以后，3个类就是3层。
        # [batch size, image_H, image_W, class_num]
        batch_hw_true = np.empty((5, 224, 224, 3))
        for i in range(5):
            pixel_class = []
            for j in range(224 * 224):
                pixel_class.append(random.randint(0, 2))
            mask = np.asarray(pixel_class).reshape((224, 224))
            mask_onehot = to_categorical(mask)
            batch_hw_true[i] = mask_onehot

        batch_hw_true = tf.constant(batch_hw_true, dtype=tf.float32)
        batch_hw_preds = tf.random_uniform(shape=[5, 224, 224, 3], minval=0, maxval=1, dtype=tf.float32)

        swcce = segmentation_weighted_categorical_crossentropy([0.5, 1, 2])
        loss4 = swcce(batch_hw_true, batch_hw_preds)
        print(sess.run(loss4))

        batch_hw_preds = batch_hw_true  # 如果pred值和true值一样，loss应该为0
        loss5 = swcce(batch_hw_true, batch_hw_preds)
        print(sess.run(loss5))
