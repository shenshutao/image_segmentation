# -*- coding: utf-8 -*-

import keras.backend as K
import tensorflow as tf


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
    # 先求batch中每个样本的cross entropy值
    batch_y_pred = tf.clip_by_value(batch_y_pred, K.epsilon(), 1 - K.epsilon())  # 防止log(0)为-inf，tf里面 0 * -inf = nan
    batch_loss = batch_y_true * tf.log(batch_y_pred) + (1 - batch_y_pred) * tf.log(1 - batch_y_pred)

    return batch_loss


def binary_focal_loss(alpha=.25, gamma=2.):
    def focal_loss(batch_y_true, batch_y_pred):
        """
        :param batch_y_true: 维度为 [batch_size]
        :param batch_y_pred: 维度为 [batch_size]
        """
        pt = tf.where(tf.equal(batch_y_true, 1), batch_y_pred, 1 - batch_y_pred)

        batch_alpha = tf.constant(alpha, shape=K.int_shape(batch_y_true), dtype=tf.float32)
        alpha_t = tf.where(tf.equal(batch_y_true, 1), batch_alpha, 1 - batch_alpha)

        # pt = tf.clip_by_value(pt, K.epsilon(), 1)  # 不会有 0 * -inf 发生，无需clip
        batch_loss = - alpha_t * (1 - pt) ** gamma * tf.log(pt)
        return batch_loss

    return focal_loss


def categorical_focal_loss(alpha=None, gamma=2.):
    """
    :param alpha: 维度为 [class_no]
    :param gamma: Float32
    """

    def focal_loss(batch_y_true, batch_y_pred):
        """
        :param batch_y_true: 维度为 [batch_size, class_no]
        :param batch_y_pred: 维度为 [batch_size, class_no]
        """
        # 归一化，加下面这段是为了兼容最后一层非Softmax的情况，如果是Softmax的输出可以注释掉，因为那个输出已经归一化了。
        batch_y_pred /= tf.reduce_sum(batch_y_pred, -1, True)

        # 防止log(0)为-inf，tf里面 0 * -inf = nan
        # 虽然也不影响back propagation，它只关心导数，不关心这个loss值。
        batch_y_pred = tf.clip_by_value(batch_y_pred, K.epsilon(), 1)
        if alpha:
            batch_loss = - tf.reduce_sum(alpha * batch_y_true * (1 - batch_y_pred) ** gamma * tf.log(batch_y_pred), -1)
        else:
            batch_loss = - tf.reduce_sum(batch_y_true * (1 - batch_y_pred) ** gamma * tf.log(batch_y_pred), -1)
        return batch_loss

    return focal_loss


def categorical_crossentropy(batch_y_true, batch_y_pred):
    """
    图片分类：
    :param batch_y_true: 维度为 [batch_size, class_no]
    :param batch_y_pred: 维度为 [batch_size, class_no]，一般为Softmax层输出的logits
    图片分割：
    :param batch_y_true: 维度为 [batch_size, image_H, image_W, class_no]
    :param batch_y_pred: 维度为 [batch_size, image_H, image_W, class_no]，一般为Softmax层输出的logits
    """
    # 归一化，加下面这段是为了兼容最后一层非Softmax的情况，如果是Softmax的输出可以注释掉，因为那个输出已经归一化了。
    batch_y_pred /= tf.reduce_sum(batch_y_pred, -1, True)

    # 先求batch中每个样本的cross entropy值
    batch_y_pred = tf.clip_by_value(batch_y_pred, K.epsilon(), 1)  # 防止log(0)为-inf，tf里面 0 * -inf = nan。
    batch_loss = - tf.reduce_sum(batch_y_true * tf.log(batch_y_pred), -1)  # 分类时为1， 分割时为3，放-1兼容两者。
    return batch_loss


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
        图片分类：
        :param batch_y_true: 维度为 [batch_size, class_no]
        :param batch_y_pred: 维度为 [batch_size, class_no]，一般为Softmax层输出的logits
        图片分割：
        :param batch_y_true: 维度为 [batch_size, image_H, image_W, class_no]
        :param batch_y_pred: 维度为 [batch_size, image_H, image_W, class_no]，一般为Softmax层输出的logits
        """
        # 归一化，加下面这段是为了兼容最后一层非Softmax的情况，如果是Softmax的输出可以注释掉，因为那个输出已经归一化了。
        batch_y_pred /= tf.reduce_sum(batch_y_pred, -1, True)

        # 先求batch中每个样本的cross entropy值
        batch_y_pred = tf.clip_by_value(batch_y_pred, K.epsilon(), 1)
        batch_loss = - tf.reduce_sum(batch_y_true * tf.log(batch_y_pred) * weights, -1)  # 分类时为1， 分割时为3，放-1兼容两者。
        return batch_loss

    return loss


