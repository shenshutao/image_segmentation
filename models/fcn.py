import tensorflow as tf
from keras import layers, models
from keras.applications import VGG16


class FCN_32s:
    def get_model(input_shape=(224, 224, 3), class_no=60):
        """
        FCN 32 模型
        :param input_shape: （输入图片长，输入图片宽，RGB层数）,注意长宽最好是32的倍数
        :param class_no: 类别数量
        :return: Keras模型
        """
        input_tensor = layers.Input(shape=input_shape)
        with tf.variable_scope("vgg_encoder"):
            encoder = VGG16(input_tensor=input_tensor, include_top=False, weights='imagenet')
            x = encoder.get_layer('block5_pool').output

        with tf.variable_scope("vgg_decoder"):
            ks = x.get_shape().as_list()[1]
            # 卷积做降采用将长宽变成1x1
            x = layers.Conv2D(filters=4096, kernel_size=(ks, ks), activation='relu', padding='valid',
                              name='fc6')(x)
            x = layers.Dropout(0.5)(x)
            x = layers.Conv2D(filters=4096, kernel_size=(1, 1), activation='relu', padding='valid', name='fc7')(x)
            x = layers.Dropout(0.5)(x)

            # 使用 1x1卷积 做卷积操作，模拟全链接层操作
            x = layers.Conv2D(filters=class_no, kernel_size=(1, 1), padding='valid')(x)

            # 使用反卷积做Upsampling
            x = layers.Deconvolution2D(filters=class_no, kernel_size=(64, 64), strides=(32, 32), padding='same',
                                       use_bias=False, name='fc8')(x)

        # 对输出的每一个像素的各类别（即各通道）的输出使用softmax
        x = layers.Activation('softmax', name='output')(x)

        model = models.Model(input=input_tensor, output=x)

        return model


class FCN_16s:
    def get_model(input_shape=(224, 224, 3), class_no=60):
        """
        FCN 16 模型
        :param input_shape: （输入图片长，输入图片宽，RGB层数）,注意长宽最好是32的倍数
        :param class_no: 类别数量
        :return: Keras模型
        """
        input_tensor = layers.Input(shape=input_shape)
        with tf.variable_scope("vgg_encoder"):
            encoder = VGG16(input_tensor=input_tensor, include_top=False, weights='imagenet')

        with tf.variable_scope("vgg_decoder"):
            x = encoder.get_layer('block5_pool').output
            ks = x.get_shape().as_list()[1]
            # 卷积做降采用将长宽变成1x1
            x = layers.Conv2D(filters=4096, kernel_size=(ks, ks), activation='relu', padding='valid',
                              name='fc6')(x)
            x = layers.Dropout(0.5)(x)
            x = layers.Conv2D(filters=4096, kernel_size=(1, 1), activation='relu', padding='valid', name='fc7')(x)
            x = layers.Dropout(0.5)(x)

            # 使用 1x1卷积 做卷积操作，模拟全链接层操作
            x = layers.Conv2D(filters=class_no, kernel_size=(1, 1), padding='valid')(x)

            # 使用反卷积做Upsampling到2x2
            x = layers.Deconvolution2D(filters=class_no, kernel_size=(4, 4), strides=(2, 2), padding='same',
                                       use_bias=False, name='upsampling')(x)

            # 拿pool4的输出
            pool4_output = encoder.get_layer('block4_pool').output
            pool4_output = layers.Conv2D(filters=class_no, kernel_size=(1, 1), padding='valid')(pool4_output)
            crop_no1 = int((pool4_output.get_shape().as_list()[1] - 2) / 2)
            pool4_output = layers.Cropping2D(cropping=((crop_no1, crop_no1), (crop_no1, crop_no1)))(pool4_output)
            x = layers.add([x, pool4_output])

            # 使用反卷积做Upsampling到32x32
            x = layers.Deconvolution2D(filters=class_no, kernel_size=(32, 32), strides=(16, 16), padding='same',
                                       use_bias=False, name='upsampling_2')(x)

        # 对输出的每一个像素的各类别（即各通道）的输出使用softmax
        x = layers.Activation('softmax', name='output')(x)

        model = models.Model(input=input_tensor, output=x)

        return model


class FCN_8s:
    def get_model(input_shape=(224, 224, 3), class_no=60):
        """
        FCN 8 模型
        :param input_shape: （输入图片长，输入图片宽，RGB层数）,注意长宽最好是32的倍数
        :param class_no: 类别数量
        :return: Keras模型
        """
        input_tensor = layers.Input(shape=input_shape)
        with tf.variable_scope("vgg_encoder"):
            encoder = VGG16(input_tensor=input_tensor, include_top=False, weights='imagenet')

        with tf.variable_scope("vgg_decoder"):
            with tf.variable_scope("fcn_32s"):
                x = encoder.get_layer('block5_pool').output  # 拿pool5的输出
                ks = x.get_shape().as_list()[1]
                # 卷积做降采用将长宽变成1x1
                x = layers.Conv2D(filters=4096, kernel_size=(ks, ks), activation='relu', padding='valid', name='fc6')(x)
                x = layers.Dropout(0.5)(x)
                x = layers.Conv2D(filters=4096, kernel_size=(1, 1), activation='relu', padding='valid', name='fc7')(x)
                x = layers.Dropout(0.5)(x)

                # 使用 1x1卷积 做卷积操作，模拟全链接层操作
                x = layers.Conv2D(filters=class_no, kernel_size=(1, 1), padding='valid')(x)

            with tf.variable_scope("fcn_16s"):
                pool4_output = encoder.get_layer('block4_pool').output  # 拿pool4的输出
                pool4_output = layers.Conv2D(filters=class_no, kernel_size=(1, 1), padding='valid')(pool4_output)

            # 使用反卷积做Upsampling到2x2
            x = layers.Deconvolution2D(filters=class_no, kernel_size=(4, 4), strides=(2, 2), padding='same',
                                       use_bias=False, name='upsampling')(x)
            # 裁剪到2x2 大小
            crop_no1 = int((pool4_output.get_shape().as_list()[1] - 2) / 2)
            pool4_output = layers.Cropping2D(cropping=((crop_no1, crop_no1), (crop_no1, crop_no1)))(pool4_output)
            x = layers.add([x, pool4_output])

            with tf.variable_scope("fcn_8s"):
                pool3_output = encoder.get_layer('block3_pool').output  # 拿pool3的输出
                pool3_output = layers.Conv2D(filters=class_no, kernel_size=(1, 1), padding='valid')(pool3_output)

            # 使用反卷积做Upsampling到4x4
            x = layers.Deconvolution2D(filters=class_no, kernel_size=(4, 4), strides=(2, 2),
                                       padding='same', use_bias=False, name='upsampling2')(x)
            # 裁剪到4x4 大小
            crop_no2 = int((pool3_output.get_shape().as_list()[1] - 4) / 2)
            pool3_output = layers.Cropping2D(cropping=((crop_no2, crop_no2), (crop_no2, crop_no2)))(pool3_output)
            x = layers.add([x, pool3_output])

            # 使用反卷积做Upsampling
            x = layers.Deconvolution2D(filters=class_no, kernel_size=(16, 16), strides=(8, 8), padding='same',
                                       use_bias=False, name='upsampling_2')(x)

        # 对输出的每一个像素的各类别（即各通道）的输出使用softmax
        x = layers.Activation('softmax', name='output')(x)

        model = models.Model(input=input_tensor, output=x)

        return model


# Test
if __name__ == "__main__":
    FCN_32s.get_model().summary()
    FCN_16s.get_model().summary()
    FCN_8s.get_model().summary()
