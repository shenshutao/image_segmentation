from custom_loss import *
import numpy as np
import random
from keras.utils import to_categorical

tf.enable_eager_execution()

if __name__ == "__main__":
    batch_bin_y_true = tf.constant([1., 1., 0., 0.])
    batch_bin_y_pred = tf.constant([1., .6, 0.1, 0.3])
    loss1 = binary_crossentropy(batch_bin_y_true, batch_bin_y_pred)
    print(loss1.numpy())

    batch_cat_y_true = tf.constant([[1., 0., 0.], [0., 1., 0.]])
    batch_cat_y_pred = tf.constant([[0., 0., 1.], [0.4, 0.1, 0.]])

    loss2 = categorical_crossentropy(batch_cat_y_true, batch_cat_y_pred)

    wcce = weighted_categorical_crossentropy([0.5, 1, 2])
    loss3 = wcce(batch_cat_y_true, batch_cat_y_pred)

    # Focal loss
    focal_loss = binary_focal_loss()
    loss6 = focal_loss(batch_bin_y_true, batch_bin_y_pred)
    print(loss6.numpy())

    batch_bin_y_true = tf.constant([1., 1., 0., 0.])
    batch_bin_y_pred = tf.constant([1., 1., 0., 0.])
    loss7 = focal_loss(batch_bin_y_true, batch_bin_y_pred)
    print(loss7.numpy())

    batch_bin_y_true = tf.constant([1., 1., 0., 0.])
    batch_bin_y_pred = tf.constant([0., 0., 1., 1.])
    loss8 = focal_loss(batch_bin_y_true, batch_bin_y_pred)
    print(loss8.numpy())

    cfl = categorical_focal_loss()
    loss9 = cfl(batch_cat_y_true, batch_cat_y_pred)

    # Segmentation
    # Batch:2 H:2 W:2 Class_no:3
    batch_hw_cat_y_true = tf.constant([[[[1., 0., 0.], [0., 1., 1.]], [[1., 0., 0.], [0., 1., 0.]]], [[[1., 0., 0.], [0., 1., 1.]], [[1., 0., 0.], [0., 1., 0.]]]])
    batch_hw_cat_y_pred = tf.constant([[[[0., 0., .2], [1., 1., .8]], [[.4, .1, .3], [.5, .7, .1]]], [[[1., 0., 0.], [0., 1., 1.]], [[1., 0., 0.], [0., 1., 0.]]]])

    logits = tf.random_uniform(shape=[5, 224, 224, 3], minval=0, maxval=1, dtype=tf.float32)

    # segmentation的标注一般是一个mask图片，每个像素一个类别，one-hot encoding以后，3个类就是3层。
    # [batch size, image_H, image_W, class num]

    batch_hw_true = np.empty((5, 224, 224, 3))
    for i in range(5):
        pixel_class = []
        for j in range(224 * 224):
            pixel_class.append(random.randint(0, 2, ))
        mask = np.asarray(pixel_class).reshape((224, 224))
        mask_onehot = to_categorical(mask)
        batch_hw_true[i] = mask_onehot

    batch_hw_true = tf.constant(batch_hw_true, dtype=tf.float32)
    batch_hw_preds = tf.random_uniform(shape=[5, 224, 224, 3], minval=0, maxval=1, dtype=tf.float32)

    # 图片语义分割
    swcce = weighted_categorical_crossentropy([0.5, 1, 2])
    loss4 = swcce(batch_hw_true, batch_hw_preds)

    batch_hw_preds = batch_hw_true  # 如果pred值和true值一样，loss应该为0
    loss5 = swcce(batch_hw_true, batch_hw_preds)
    print(loss5.numpy())
    print(loss5.numpy().shape)

    cfl = categorical_focal_loss()
    loss9 = cfl(batch_hw_true, batch_hw_preds)
    print(loss9.numpy().shape)