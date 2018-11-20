from keras.applications import Xception
from keras import layers, models
import tensorflow as tf

class DeepLabV3Plus:
    def get_model(input_shape = (224, 224, 3), class_no = 21):
        # 先做一个用原始Xception的版本，因为现在找不到改进版Xception的pretrain weights，之后训练一个
        input_tensor = layers.Input(shape=input_shape)
        with tf.variable_scope("encoder"):
            encoder = Xception(input_tensor=input_tensor, include_top=False, weights='imagenet')
            x = encoder.output
        # with tf.variable_scope("decoder"):
        #TODO



        model = models.Model(inputs=input_tensor, outputs=x)

        return model


# Test
if __name__ == "__main__":
    DeepLabV3Plus.get_model(input_shape=(224, 224, 3)).summary()