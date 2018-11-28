## Briefing
This is a image semantic segmentation demo using Keras.   
To simplify the code, I choose the horse dataset, as the two classes are quite balanced (background and horse).      

## Horse Dataset
Horse dataset is downloaded from http://www.msri.org/people/members/eranb/
 
## Nets
1. FCN, https://arxiv.org/abs/1411.4038, translated from the original caffe code https://github.com/shelhamer/fcn.berkeleyvision.org 
2. Unet, https://arxiv.org/abs/1505.04597    
3. DeepLab V3+ (onging), https://arxiv.org/abs/1802.02611

## Loss
Inside the custom_loss.py, there are some losses not only for segmentation task, also for binary classification or category classification.
Some famous loss implemented, such as focal loss.

The custom_loss_eagermode.py is only for loss function testing purpose, testing on eager mode is more efficient.   

## Class imbalance
However the class imbalance is always a big problem in daily segmentation tasks.     
Tried on Pascal dataset, but the result is bad, still exploring.    
1. Weight cross entropy, what's the reasonable loss weights for classes ? (the inverse class frequency? ongoing)
2. Dice loss / GDL (onging)
3. Tversky loss  (onging)


