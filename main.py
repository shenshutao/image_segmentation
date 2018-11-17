import sys

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.models import load_model

from custom_loss import *
from custom_metrics import *
from data_gens import *
from models import *

sys.setrecursionlimit(10000)

if __name__ == "__main__":
    # Use VOC 2012 Dataset
    horse_path = 'weizmann_horse_db'
    batch_size = 32

    train_gen = horse_gen.get_horse_generator(horse_path, train_or_val='train', batch_size=batch_size,
                                              input_hw=(224, 224, 3), mask_hw=(224, 224, 2))

    # model = FCN.get_fcn8s_model(input_shape=(224, 224, 3), class_no=2)
    # model = FCN.get_fcn16s_model(input_shape=(224, 224, 3), class_no=2)
    model = FCN.get_fcn32s_model(input_shape=(224, 224, 3), class_no=2)
    # model = Unet.get_unet_model(input_shape=(224, 224, 3), class_no=2)

    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[mean_iou(num_class=2)])
    model.compile(loss=categorical_crossentropy, optimizer='adam', metrics=[mean_iou(num_class=2)])

    model.summary()

    checkpoint = ModelCheckpoint('fcn16s.h5', verbose=1, save_best_only=False, period=3)
    tensor_board = TensorBoard(log_dir='log', histogram_freq=0, write_graph=True, write_grads=True, write_images=True)
    learning_rate_reduction = ReduceLROnPlateau(monitor='loss', patience=2, verbose=1, factor=0.5, min_lr=0.000001)

    model.fit_generator(
        train_gen,
        steps_per_epoch=100,
        epochs=5,
        callbacks=[checkpoint, tensor_board, learning_rate_reduction]
    )

    model.save('fcn16s.h5')

    print('Start Test')
    model = load_model('fcn16s.h5', compile=False)
    # 取val集10张图片，测试一下效果
    val_gen = horse_gen.get_horse_generator(horse_path, 'val', 1, input_hw=(224, 224, 3), mask_hw=(224, 224, 2))

    i = 0
    for val_images, mask in val_gen:
        img_np = val_images[0]
        img_np = (img_np + 1.) * 127.5
        im0 = Image.fromarray(np.uint8(img_np))
        im0.save('output/{}_img.jpg'.format(i))

        res = model.predict(val_images)[0]
        pred_label = res.argmax(axis=2)
        pred_label[pred_label == 1] = 255
        im1 = Image.fromarray(np.uint8(pred_label))
        im1.save('output/{}_pred.png'.format(i))

        true_label = mask[0].argmax(axis=2)
        true_label[true_label == 1] = 255
        im2 = Image.fromarray(np.uint8(true_label))
        im2.save('output/{}_true.png'.format(i))

        i += 1
        if i == 100:
            print('End test')
            exit(1)
