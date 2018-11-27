from keras.applications import Xception
from keras import layers, models
import tensorflow as tf
import keras.backend as K
import numpy as np


# Only for input_size (513, 513) atrous_rates = [6, 12, 18] and output_stride = 16
class Xception_Adv:
    def separable_conv2d_with_bn(x, prefix, filters, kernel_size=(3, 3), strides=(1, 1),
                                 activation_fn_in_separable_conv=False):
        """
        Add Batch Norm Layer between Depthwise Conv & Pointwise Conv layer
        """
        x = layers.DepthwiseConv2D(kernel_size, strides=strides, padding='same', use_bias=False,
                                   name=prefix + '_depthwise')(x)
        x = layers.BatchNormalization(name=prefix + '_depthwise_bn')(x)
        if activation_fn_in_separable_conv:
            x = layers.Activations('relu')(x)
        x = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False, name=prefix + '_pointwise')(x)
        x = layers.BatchNormalization(name=prefix + '_pointwise_bn')(x)

        return x

    def xception_moudle(x, prefix, depth_list, skip_connection_type='conv', activation_fn_in_separable_conv=False,
                        ds_strides=(1, 1)):
        # Prepare shortcut
        if skip_connection_type == 'conv':
            residual = layers.Conv2D(depth_list[2], (1, 1), strides=ds_strides, padding='same', use_bias=False)(x)
            residual = layers.BatchNormalization()(residual)
        elif skip_connection_type == 'sum':
            residual = x

        x = layers.Activation('relu')(x)
        x = Xception_Adv.separable_conv2d_with_bn(x, prefix + '_c1', depth_list[0],
                                                  activation_fn_in_separable_conv=activation_fn_in_separable_conv)
        x = layers.Activation('relu')(x)
        x = Xception_Adv.separable_conv2d_with_bn(x, prefix + '_c2', depth_list[1],
                                                  activation_fn_in_separable_conv=activation_fn_in_separable_conv)
        x = layers.Activation('relu')(x)
        x = Xception_Adv.separable_conv2d_with_bn(x, prefix + '_c3', depth_list[2], strides=ds_strides,
                                                  activation_fn_in_separable_conv=activation_fn_in_separable_conv)
        if skip_connection_type in ('conv', 'sum'):
            x = layers.add([x, residual])
        return x

    def get_enhanced_xception(input_tensor):
        img_input = input_tensor

        with tf.variable_scope("Xception_65"):
            with tf.variable_scope("entry_flow"):
                with tf.variable_scope("conv_1_1"):
                    x = layers.Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, padding='same', name='entry_conv_1')(
                        img_input)
                    x = layers.BatchNormalization()(x)
                    x = layers.Activation('relu')(x)

                with tf.variable_scope("conv_1_2"):
                    x = layers.Conv2D(64, (3, 3), use_bias=False, padding='same', name='entry_conv_2')(x)
                    x = layers.BatchNormalization()(x)
                    x = layers.Activation('relu')(x)

                with tf.variable_scope("block_1"):
                    x = Xception_Adv.xception_moudle(x, prefix='entry_block1', depth_list=(128, 128, 128), skip_connection_type='conv', ds_strides=(2, 2))

                with tf.variable_scope("block_2"):
                    x = Xception_Adv.xception_moudle(x, prefix='entry_block2', depth_list=(256, 256, 256), skip_connection_type='conv', ds_strides=(2, 2))

                with tf.variable_scope("block_3"):
                    x = Xception_Adv.xception_moudle(x, prefix='entry_block3', depth_list=(728, 728, 728), skip_connection_type='conv', ds_strides=(2, 2))

            with tf.variable_scope("middle_flow"):
                for i in range(16):
                    prefix = 'unit_' + str(i + 1)
                    with tf.variable_scope(prefix):
                        x = Xception_Adv.xception_moudle(x, prefix=prefix, depth_list=(728, 728, 728),
                                                         skip_connection_type='sum', ds_strides=(1, 1))

            with tf.variable_scope("exit_flow"):
                with tf.variable_scope('block_1'):
                    x = Xception_Adv.xception_moudle(x, prefix='exit_b1', depth_list=(728, 1024, 1024),
                                                     skip_connection_type='conv', ds_strides=(1, 1))  # hit the limit, stride 16, so no stride here, keep the dimension as 33x33

                with tf.variable_scope('block_2'):
                    x = Xception_Adv.xception_moudle(x, prefix='exit_b2', depth_list=(1536, 1536, 2048),
                                                     skip_connection_type='none', ds_strides=(1, 1))

        model = models.Model(inputs=img_input, outputs=x, name='enhanced_xception')

        return model


class DeepLabV3Plus:

    def get_atrous_conv(x_output, atrous_rate=(6, 12, 18)):
        # 1x1 Conv
        aspp0 = layers.Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x_output)
        aspp0 = layers.BatchNormalization(name='aspp0_BN', epsilon=1e-5)(aspp0)
        aspp0 = layers.Activation('relu', name='aspp0_activation')(aspp0)

        # 3x3 Conv rate 6
        aspp1 = layers.Conv2D(256, (3, 3), dilation_rate=(atrous_rate[0], atrous_rate[0]), padding='same')(x_output)
        aspp1 = layers.BatchNormalization(epsilon=1e-5)(aspp1)
        aspp1 = layers.Activation('relu')(aspp1)

        # 3x3 Conv rate 12
        aspp2 = layers.Conv2D(256, (3, 3), dilation_rate=(atrous_rate[1], atrous_rate[1]), padding='same')(x_output)
        aspp2 = layers.BatchNormalization(epsilon=1e-5)(aspp2)
        aspp2 = layers.Activation('relu')(aspp2)

        # 3x3 Conv rate 18
        aspp3 = layers.Conv2D(256, (3, 3), dilation_rate=(atrous_rate[2], atrous_rate[2]), padding='same')(x_output)
        aspp3 = layers.BatchNormalization(epsilon=1e-5)(aspp3)
        aspp3 = layers.Activation('relu')(aspp3)

        # Image Pooling
        aspp4 = layers.GlobalAveragePooling2D()(x_output)
        aspp4 = layers.Reshape((1, 1, -1))(aspp4)
        aspp4 = layers.Conv2D(256, (1, 1), padding='same', use_bias=False, name='image_pooling')(aspp4)  # 减少层数
        aspp4 = layers.Activation('relu')(aspp4)
        aspp4 = layers.Lambda(
            lambda s: tf.image.resize_bilinear(s, size=(K.int_shape(x_output)[1], K.int_shape(x_output)[2]),
                                               name='UpSampling_aspp4'))(aspp4)  # Reshape back for concat

        x = layers.Concatenate()([aspp0, aspp1, aspp2, aspp3, aspp4])

        return x

    # 比普通的Conv2D少了很多参数
    def get_separable_atrous_conv(x_output, atrous_rate=(6, 12, 18)):
        # 1x1 Conv
        aspp0 = layers.Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x_output)
        aspp0 = layers.BatchNormalization(name='aspp0_BN', epsilon=1e-5)(aspp0)
        aspp0 = layers.Activation('relu', name='aspp0_activation')(aspp0)

        # 3x3 Separable Conv rate 6
        aspp1 = layers.DepthwiseConv2D((3, 3), dilation_rate=(atrous_rate[0], atrous_rate[0]), padding='same',
                                       use_bias=False, name='aspp1_depthwise')(x_output)
        aspp1 = layers.BatchNormalization()(aspp1)
        aspp1 = layers.Activation('relu')(aspp1)
        aspp1 = layers.Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp1_pointwise')(aspp1)
        aspp1 = layers.BatchNormalization(epsilon=1e-5)(aspp1)
        aspp1 = layers.Activation('relu')(aspp1)

        # 3x3 Separable Conv rate 12
        aspp2 = layers.DepthwiseConv2D((3, 3), dilation_rate=(atrous_rate[1], atrous_rate[1]), padding='same',
                                       use_bias=False, name='aspp2_depthwise')(x_output)
        aspp2 = layers.BatchNormalization()(aspp2)
        aspp2 = layers.Activation('relu')(aspp2)
        aspp2 = layers.Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp2_pointwise')(aspp2)
        aspp2 = layers.BatchNormalization(epsilon=1e-5)(aspp2)
        aspp2 = layers.Activation('relu')(aspp2)

        # 3x3 Separable Conv rate 18
        aspp3 = layers.DepthwiseConv2D((3, 3), dilation_rate=(atrous_rate[2], atrous_rate[2]), padding='same',
                                       use_bias=False, name='aspp3_depthwise')(x_output)
        aspp3 = layers.BatchNormalization()(aspp3)
        aspp3 = layers.Activation('relu')(aspp3)
        aspp3 = layers.Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp3_pointwise')(aspp3)
        aspp3 = layers.BatchNormalization(epsilon=1e-5)(aspp3)
        aspp2 = layers.Activation('relu')(aspp2)

        # Image Pooling
        aspp4 = layers.GlobalAveragePooling2D()(x_output)
        aspp4 = layers.Reshape((1, 1, -1))(aspp4)
        aspp4 = layers.Conv2D(256, (1, 1), padding='same', use_bias=False, name='image_pooling')(aspp4)  # 减少层数
        aspp4 = layers.Activation('relu')(aspp4)
        aspp4 = layers.Lambda(
            lambda s: tf.image.resize_bilinear(s, size=(K.int_shape(x_output)[1], K.int_shape(x_output)[2]),
                                               name='UpSampling_aspp4'))(aspp4)  # Reshape back for concat

        x = layers.Concatenate()([aspp0, aspp1, aspp2, aspp3, aspp4])
        return x

    def get_model(input_shape=(513, 513, 3), class_no=21):
        input_tensor = layers.Input(shape=input_shape)
        with tf.variable_scope("encoder"):
            encoder = Xception_Adv.get_enhanced_xception(input_tensor=input_tensor)
            x_output = encoder.output

            # for layer in encoder.layers:  #  not available as pre train model is not ready here.
            #     layer.trainable = False

            # 1x1 Conv, Use dilation rate (6, 12, 18), as the output H x W is 33 x 33
            x = DeepLabV3Plus.get_separable_atrous_conv(x_output, atrous_rate=(6, 12, 18))

            x = layers.Conv2D(256, (1, 1), padding='same', use_bias=False, name='concat_projection')(x)
            x = layers.BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
            x = layers.Activation('relu')(x)
            x = layers.Dropout(0.1)(x)

        with tf.variable_scope("decoder"):
            # x4 (x2) block
            x = layers.Lambda(lambda s: tf.image.resize_bilinear(s, size=(
                int(np.ceil(input_shape[0] / 4)), int(np.ceil(input_shape[0] / 4)))), name='UpSampling1')(x)

            skip1 = encoder.get_layer('entry_block2_c2_pointwise_bn').output

            dec_skip1 = layers.Conv2D(48, (1, 1), padding='same', use_bias=False, name='feature_projection0')(skip1)
            dec_skip1 = layers.BatchNormalization(name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
            dec_skip1 = layers.Activation('relu')(dec_skip1)
            x = layers.Concatenate()([x, dec_skip1])
            x = layers.Conv2D(304, (3, 3), padding='same')(x)

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
