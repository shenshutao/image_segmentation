import sys

from PIL import Image
import os
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from custom_loss import *
from custom_metrics import *
from models import *
from data_gens import *
from keras.models import load_model
import os

sys.setrecursionlimit(10000)

if __name__ == "__main__":
    # Use VOC 2012 Dataset
    image_list = [os.path.join('D:/Datasets/testImage/PNGImages',  image) for image in os.listdir('D:/Datasets/testImage/PNGImages')]
    mask_list = [os.path.join('D:/Datasets/testImage/SegmentationClassRaw', mask) for mask in os.listdir('D:/Datasets/testImage/SegmentationClassRaw')]

    batch_size = 8
    train_gen = normal_gen.get_normal_generator(image_list[:-20], mask_list[:-20], batch_size=batch_size,
                                              input_hw=(512, 512, 3), mask_hw=(512, 512, 4))

    # model = FCN.get_fcn16s_model(input_shape=(512, 512, 3), class_no=4)
    model = Unet.get_unet_model(input_shape=(512, 512, 3), class_no=4)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[mean_iou(num_class=4)])
    model.summary()

    checkpoint = ModelCheckpoint('zg.h5', verbose=1, save_best_only=False, period=3)
    tensor_board = TensorBoard(log_dir='log', histogram_freq=0, write_graph=True, write_grads=True, write_images=True)
    learning_rate_reduction = ReduceLROnPlateau(monitor='loss', patience=5, verbose=1, factor=0.5, min_lr=0.000001)

    model.fit_generator(
        train_gen,
        steps_per_epoch=100,
        epochs=500,
        callbacks=[checkpoint, tensor_board, learning_rate_reduction]
    )

    model.save('zg.h5')

    print('Start Test')
    model = load_model('zg.h5', compile=False)
    # 取val集10张图片，测试一下效果
    val_gen =  normal_gen.get_normal_generator(image_list[-20:], mask_list[-20:], batch_size=batch_size,
                                              input_hw=(512, 512, 3), mask_hw=(512, 512, 4))

    i = 0
    for val_images, mask in val_gen:
        img_np = val_images[0]
        img_np = (img_np + 1.) * 127.5
        im0 = Image.fromarray(np.uint8(img_np))
        im0.save('output/{}_img.jpg'.format(i))

        res = model.predict(val_images)[0]
        pred_label = res.argmax(axis=2)
        pred_label = pred_label * 60
        im1 = Image.fromarray(np.uint8(pred_label))
        im1.save('output/{}_pred.png'.format(i))

        true_label = mask[0].argmax(axis=2)
        true_label = true_label * 60
        im2 = Image.fromarray(np.uint8(true_label))
        im2.save('output/{}_true.png'.format(i))

        i += 1
        if i == 100:
            print('End test')
            exit(1)
