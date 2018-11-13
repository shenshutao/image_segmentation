import tensorflow as tf
from keras import layers, models
from keras.applications import VGG16
import keras.backend as K
import keras

class FCN:
    def center_crop(original_ts, target_ts):
        crop_no = (K.int_shape(original_ts)[1] - K.int_shape(target_ts)[1]) // 2
        if (K.int_shape(original_ts)[1] - K.int_shape(target_ts)[1]) % 2 == 0:
            croped_ts = layers.Cropping2D(cropping=((crop_no, crop_no), (crop_no, crop_no)))(original_ts)
        else:
            croped_ts = layers.Cropping2D(cropping=((crop_no + 1, crop_no), (crop_no + 1, crop_no)))(original_ts)

        return croped_ts

    def get_fcn32s_model(input_shape=(224, 224, 3), class_no=21):
        """
        FCN 32 模型
        :param input_shape: （输入图片长，输入图片宽，RGB层数）,注意长宽最好是32的倍数
        :param class_no: 类别数量
        :return: Keras模型
        """
        input_tensor = layers.Input(shape=input_shape)
        x = layers.ZeroPadding2D(padding=(99, 99))(input_tensor)  # Pad 100, 99 + 1 in first layer of vgg
        with tf.variable_scope("vgg_encoder"):
            encoder = VGG16(input_tensor=x, include_top=False, weights='imagenet')

        with tf.variable_scope("vgg_decoder"):
            x = encoder.get_layer('block5_pool').output
            # 卷积做降采用
            x = layers.Conv2D(filters=4096, kernel_size=(7, 7), activation='relu', padding='valid', name='fc6')(x)
            x = layers.Dropout(0.5)(x)
            x = layers.Conv2D(filters=4096, kernel_size=(1, 1), activation='relu', padding='valid', name='fc7')(x)
            x = layers.Dropout(0.5)(x)

            # 使用 1x1卷积 做卷积操作，模拟全链接层操作
            x = layers.Conv2D(filters=class_no, kernel_size=(1, 1), padding='valid')(x)

            # 使用反卷积做Upsampling
            x = layers.Conv2DTranspose(filters=class_no, kernel_size=(64, 64), strides=(32, 32), padding='same',
                                       use_bias=False, name='Upsampling1')(x)

        # 如果size不够，再做一个Bilinear的Upsampling（通常在图片size不为32的倍数时候需要）
        if K.int_shape(x)[1:3] != K.int_shape(input_tensor)[1:3]:
            print('Size different, do Bilinear Upsampling')
            x = layers.Lambda(lambda x: tf.image.resize_bilinear(x, size=K.int_shape(input_tensor)[1:3]))(x)

        # 对输出的每一个像素的各类别（即各通道）的输出使用softmax
        x = layers.Activation('softmax', name='output')(x)

        model = models.Model(inputs=input_tensor, outputs=x)

        return model

    def get_fcn16s_model(input_shape=(224, 224, 3), class_no=21):
        """
        FCN 16 模型
        :param input_shape: （输入图片长，输入图片宽，RGB层数）,注意长宽最好是32的倍数
        :param class_no: 类别数量
        :return: Keras模型
        """
        input_tensor = layers.Input(shape=input_shape)
        x = layers.ZeroPadding2D(padding=(99, 99))(input_tensor)  # Pad 100, 99 + 1 in first layer of vgg
        with tf.variable_scope("vgg_encoder"):
            encoder = VGG16(input_tensor=x, include_top=False, weights='imagenet')

        with tf.variable_scope("vgg_decoder"):
            with tf.variable_scope("fcn_32s"):
                x = encoder.get_layer('block5_pool').output
                # 卷积做降采用
                x = layers.Conv2D(filters=4096, kernel_size=(7, 7), activation='relu', padding='valid', name='fc6')(x)
                x = layers.Dropout(0.5)(x)
                x = layers.Conv2D(filters=4096, kernel_size=(1, 1), activation='relu', padding='valid', name='fc7')(x)
                x = layers.Dropout(0.5)(x)

                # 使用 1x1卷积 做卷积操作，模拟全链接层操作
                x = layers.Conv2D(filters=class_no, kernel_size=(1, 1), padding='valid')(x)

            # 使用反卷积做Upsampling到2倍
            x = layers.Conv2DTranspose(filters=class_no, kernel_size=(4, 4), strides=(2, 2), padding='same',
                                       use_bias=False, name='upsampling1')(x)

            # print(tf.size_shape(x))
            with tf.variable_scope("fcn_16s"):
                # 拿pool4的输出
                pool4_output = encoder.get_layer('block4_pool').output
                pool4_output = layers.Conv2D(filters=class_no, kernel_size=(1, 1), padding='valid')(pool4_output)

            pool4_crop = FCN.center_crop(pool4_output, x)
            x = layers.add([x, pool4_crop])

            # 使用反卷积做Upsampling到32倍
            x = layers.Conv2DTranspose(filters=class_no, kernel_size=(32, 32), strides=(16, 16), padding='same',
                                       use_bias=False, name='upsampling2')(x)

        # 如果size不够，再做一个Bilinear的Upsampling（通常在图片size不为32的倍数时候需要）
        if K.int_shape(x)[1:3] != K.int_shape(input_tensor)[1:3]:
            print('Size different, do Bilinear Upsampling')
            x = layers.Lambda(lambda x: tf.image.resize_bilinear(x, size=K.int_shape(input_tensor)[1:3]))(x)

        # 对输出的每一个像素的各类别（即各通道）的输出使用softmax
        x = layers.Activation('softmax', name='output')(x)

        model = models.Model(inputs=input_tensor, outputs=x)

        return model

    def get_fcn8s_model(input_shape=(224, 224, 3), class_no=21):
        """
        FCN 8 模型
        :param input_shape: （输入图片长，输入图片宽，RGB层数）,注意长宽最好是32的倍数
        :param class_no: 类别数量
        :return: Keras模型
        """
        input_tensor = layers.Input(shape=input_shape)
        x = layers.ZeroPadding2D(padding=(99, 99))(input_tensor)  # Pad 100, 99 + 1 in first layer of vgg
        with tf.variable_scope("vgg_encoder"):
            encoder = VGG16(input_tensor=x, include_top=False, weights='imagenet')

        with tf.variable_scope("vgg_decoder"):
            with tf.variable_scope("fcn_32s"):
                x = encoder.get_layer('block5_pool').output  # 拿pool5的输出
                # 卷积做降采用
                x = layers.Conv2D(filters=4096, kernel_size=(7, 7), activation='relu', padding='valid', name='fc6')(x)
                x = layers.Dropout(0.5)(x)
                x = layers.Conv2D(filters=4096, kernel_size=(1, 1), activation='relu', padding='valid', name='fc7')(x)
                x = layers.Dropout(0.5)(x)

                # 使用 1x1卷积 做卷积操作，模拟全链接层操作
                x = layers.Conv2D(filters=class_no, kernel_size=(1, 1), padding='valid')(x)

            # 使用反卷积做Upsampling到2倍
            x = layers.Conv2DTranspose(filters=class_no, kernel_size=(4, 4), strides=(2, 2), padding='same',
                                       use_bias=False, name='upsampling1')(x)

            with tf.variable_scope("fcn_16s"):
                pool4_output = encoder.get_layer('block4_pool').output  # 拿pool4的输出
                pool4_output = layers.Conv2D(filters=class_no, kernel_size=(1, 1), padding='valid')(pool4_output)

            # 裁剪到2x2 大小
            pool4_crop = FCN.center_crop(pool4_output, x)
            x = layers.add([x, pool4_crop])

            with tf.variable_scope("fcn_8s"):
                pool3_output = encoder.get_layer('block3_pool').output  # 拿pool3的输出
                pool3_output = layers.Conv2D(filters=class_no, kernel_size=(1, 1), padding='valid')(pool3_output)

            # 使用反卷积做Upsampling到4倍
            x = layers.Conv2DTranspose(filters=class_no, kernel_size=(4, 4), strides=(2, 2),
                                       padding='same', use_bias=False, name='upsampling2')(x)
            # 中心裁剪
            pool3_crop = FCN.center_crop(pool3_output, x)
            x = layers.add([x, pool3_crop])

            # 使用反卷积做Upsampling
            x = layers.Conv2DTranspose(filters=class_no, kernel_size=(16, 16), strides=(8, 8), padding='same',
                                       use_bias=False, name='upsampling3')(x)

        # 如果size不够，再做一个Bilinear的Upsampling（通常在图片size不为32的倍数时候需要）
        if K.int_shape(x)[1:3] != K.int_shape(input_tensor)[1:3]:
            print('Size different, do Bilinear Upsampling')
            x = layers.Lambda(lambda x: tf.image.resize_bilinear(x, size=K.int_shape(input_tensor)[1:3]))(x)

        # 对输出的每一个像素的各类别（即各通道）的输出使用softmax
        x = layers.Activation('softmax', name='output')(x)

        model = models.Model(inputs=input_tensor, outputs=x)

        return model


# Test
if __name__ == "__main__":
    FCN.get_fcn32s_model(input_shape=(500, 500, 3)).summary()
    FCN.get_fcn16s_model(input_shape=(500, 500, 3)).summary()
    FCN.get_fcn8s_model(input_shape=(500, 500, 3)).summary()
