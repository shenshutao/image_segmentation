from keras.applications import Xception
from keras import layers, models
import tensorflow as tf
import keras.backend as K
import numpy as np


# class DepthwiseSeparableConv(Layer):


class Xception_Adv:
    def get_enhanced_xception(input_tensor):
        img_input = input_tensor

        with tf.variable_scope("Xception_65"):
            # x = layers.ZeroPadding2D(padding=(1, 1))(img_input)
        ##### Entry_flow #####
            with tf.variable_scope("entry_flow"):
                with tf.variable_scope("conv_1_1"):
                    x = layers.Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, padding='same', name='conv_1_1')(img_input)
                    x = layers.BatchNormalization(name='conv_1_1_bn')(x)
                    x = layers.Activation('relu', name='conv_1_1_act')(x)

                with tf.variable_scope("conv_1_2"):
                    x = layers.Conv2D(64, (3, 3), use_bias=False, padding='same', name='conv_1_2')(x)
                    x = layers.BatchNormalization(name='conv_1_2_bn')(x)
                    x = layers.Activation('relu', name='conv_1_2_act')(x)

                with tf.variable_scope("block_1"):
                    # 开始3个
                    residual = layers.Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
                    residual = layers.BatchNormalization()(residual)

                    x = layers.SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block_1_sepconv1')(x)
                    x = layers.BatchNormalization(name='block_1_sepconv1_bn')(x)
                    x = layers.Activation('relu', name='block_1_sepconv2_act')(x)
                    x = layers.SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block_1_sepconv2')(x)
                    x = layers.BatchNormalization(name='block_1_sepconv2_bn')(x)

                    # x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block_1_pool')(x)
                    x = layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', use_bias=False, name='block_1_depthwise')(x)
                    x = layers.BatchNormalization(name='block_1_depthwise_bn')(x)
                    x = layers.Activation('relu')(x)
                    x = layers.Conv2D(128, (1, 1), padding='same', use_bias=False, name='block_1_pointwise')(x)
                    x = layers.BatchNormalization(name='block_1_pointwise_bn')(x)
                    x = layers.Activation('relu')(x)
                    # x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', use_bias=False, name='block_1_alt')(x) # 假如用这个，参数多了好几倍

                    x = layers.add([x, residual])

                with tf.variable_scope("block_2"):
                    residual = layers.Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
                    residual = layers.BatchNormalization()(residual)

                    x = layers.Activation('relu', name='block_2_sepconv1_act')(x)
                    x = layers.SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block_2_sepconv1')(x)
                    x = layers.BatchNormalization(name='block_2_sepconv1_bn')(x)
                    x = layers.Activation('relu', name='block_2_sepconv2_act')(x)
                    x = layers.SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block_2_sepconv2')(x)
                    x = layers.BatchNormalization(name='block_2_sepconv2_bn')(x)

                    # x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block_2_pool')(x)
                    x = layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', use_bias=False, name='block_2_depthwise')(x)
                    x = layers.BatchNormalization(name='block_2_depthwise_bn')(x)
                    x = layers.Activation('relu')(x)
                    x = layers.Conv2D(256, (1, 1), padding='same', use_bias=False, name='block_2_pointwise')(x)
                    x = layers.BatchNormalization(name='block_2_pointwise_bn')(x)
                    x = layers.Activation('relu')(x)

                    x = layers.add([x, residual])

                with tf.variable_scope("block_3"):
                    residual = layers.Conv2D(728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
                    residual = layers.BatchNormalization()(residual)

                    x = layers.Activation('relu', name='block_3_sepconv1_act')(x)
                    x = layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block_3_sepconv1')(x)
                    x = layers.BatchNormalization(name='block_3_sepconv1_bn')(x)
                    x = layers.Activation('relu', name='block_3_sepconv2_act')(x)
                    x = layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block_3_sepconv2')(x)
                    x = layers.BatchNormalization(name='block_3_sepconv2_bn')(x)

                    # x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block_3_pool')(x)
                    x = layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', use_bias=False, name='block_3_depthwise')(x)
                    x = layers.BatchNormalization(name='block_3_depthwise_bn')(x)
                    x = layers.Activation('relu')(x)
                    x = layers.Conv2D(728, (1, 1), padding='same', use_bias=False, name='block_3_pointwise')(x)
                    x = layers.BatchNormalization(name='block_3_pointwise_bn')(x)
                    x = layers.Activation('relu')(x)

                    x = layers.add([x, residual])

            with tf.variable_scope("middle_flow"):
                for i in range(16):
                    residual = x
                    prefix = 'unit_' + str(i+1)

                    x = layers.Activation('relu', name=prefix + '_sepconv1_act')(x)
                    x = layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(x)
                    x = layers.BatchNormalization(name=prefix + '_sepconv1_bn')(x)
                    x = layers.Activation('relu', name=prefix + '_sepconv2_act')(x)
                    x = layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(x)
                    x = layers.BatchNormalization(name=prefix + '_sepconv2_bn')(x)
                    x = layers.Activation('relu', name=prefix + '_sepconv3_act')(x)
                    x = layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(x)
                    x = layers.BatchNormalization(name=prefix + '_sepconv3_bn')(x)

                    x = layers.add([x, residual])

            with tf.variable_scope("exit_flow"):
                residual = layers.Conv2D(1024, (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
                residual = layers.BatchNormalization()(residual)

                x = layers.Activation('relu')(x)
                x = layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
                x = layers.BatchNormalization()(x)
                x = layers.Activation('relu')(x)
                x = layers.SeparableConv2D(1024, (3, 3), padding='same', use_bias=False)(x)
                x = layers.BatchNormalization()(x)

                x = layers.BatchNormalization()(x)
                x = layers.Activation('relu')(x)
                x = layers.Conv2D(1024, (1, 1), padding='same', use_bias=False)(x)
                x = layers.BatchNormalization()(x)
                x = layers.Activation('relu')(x)
                x = layers.add([x, residual])

                x = layers.SeparableConv2D(1536, (3, 3), padding='same', use_bias=False, name='block22_sepconv1')(x)
                x = layers.BatchNormalization(name='block22_sepconv1_bn')(x)
                x = layers.Activation('relu', name='block22_sepconv1_act')(x)

                x = layers.SeparableConv2D(2048, (3, 3), padding='same', use_bias=False, name='block22_sepconv2')(x)
                x = layers.BatchNormalization(name='block22_sepconv2_bn')(x)
                x = layers.Activation('relu', name='block22_sepconv2_act')(x)

        model = models.Model(inputs=img_input, outputs=x, name='enhanced_xception')

        return model


class DeepLabV3Plus:
    def get_model(input_shape=(513, 513, 3), class_no=21):
        input_tensor = layers.Input(shape=input_shape)
        with tf.variable_scope("encoder"):
            encoder = Xception_Adv.get_enhanced_xception(input_tensor=input_tensor)
            x_output = encoder.output

            # for layer in encoder.layers:
            #     layer.trainable = False

            # 1x1 Conv
            aspp0 = layers.Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x_output)
            aspp0 = layers.BatchNormalization(name='aspp0_BN', epsilon=1e-5)(aspp0)
            aspp0 = layers.Activation('relu', name='aspp0_activation')(aspp0)

            aspp1 = layers.Conv2D(filters=256, kernel_size=(3, 3), dilation_rate=(6, 6), padding='same')(x_output)
            aspp1 = layers.BatchNormalization(epsilon=1e-5)(aspp1)
            aspp1 = layers.Activation('relu')(aspp1)

            aspp2 = layers.Conv2D(filters=256, kernel_size=(3, 3), dilation_rate=(12, 12), padding='same')(x_output)
            aspp2 = layers.BatchNormalization(epsilon=1e-5)(aspp2)
            aspp2 = layers.Activation('relu')(aspp2)

            aspp3 = layers.Conv2D(filters=256, kernel_size=(3, 3), dilation_rate=(18, 18), padding='same')(x_output)
            aspp3 = layers.BatchNormalization(epsilon=1e-5)(aspp3)
            aspp3 = layers.Activation('relu')(aspp3)

            aspp4 = layers.GlobalAveragePooling2D()(x_output)
            aspp4 = layers.Reshape((1, 1, -1))(aspp4)
            aspp4 = layers.Conv2D(256, (1, 1), padding='same', use_bias=False, name='image_pooling')(aspp4)  # 减少层数
            aspp4 = layers.Lambda(
                lambda s: tf.image.resize_bilinear(s, size=(K.int_shape(x_output)[1], K.int_shape(x_output)[2]),
                                                   name='UpSampling_aspp4'))(aspp4)  # Reshape back for concat

            x = layers.Concatenate()([aspp0, aspp1, aspp2, aspp3, aspp4])

            x = layers.Conv2D(256, (1, 1), padding='same', use_bias=False, name='concat_projection')(x)
            x = layers.BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
            x = layers.Activation('relu')(x)
            x = layers.Dropout(0.1)(x)

        with tf.variable_scope("decoder"):
            # x4 (x2) block
            x = layers.Lambda(lambda s: tf.image.resize_bilinear(s, size=(
                int(np.ceil(input_shape[0] / 4)), int(np.ceil(input_shape[0] / 4)))), name='UpSampling1')(x)

            dec_skip1 = layers.Conv2D(48, (1, 1), padding='same', use_bias=False, name='feature_projection0')(
                encoder.get_layer('block_2_sepconv2_bn').output)
            dec_skip1 = layers.BatchNormalization(name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
            dec_skip1 = layers.Activation('relu')(dec_skip1)
            x = layers.Concatenate()([x, dec_skip1])
            x = layers.Conv2D(filters=304, kernel_size=(3, 3), padding='same')(x)

            x = layers.Conv2D(class_no, (1, 1), padding='same')(x)
            x = layers.Lambda(lambda s: tf.image.resize_bilinear(s, size=K.int_shape(input_tensor)[1:3]),
                              name='UpSampling2')(x)

        model = models.Model(inputs=input_tensor, outputs=x, name='deeplab_try')

        return model


# Test
if __name__ == "__main__":
    # input_tensor = layers.Input(shape=(513, 513, 3))
    # Xception_Adv.get_enhanced_xception(input_tensor).summary()
    #
    # Xception(input_tensor=input_tensor, include_top=False).summary()

    DeepLabV3Plus.get_model().summary()
