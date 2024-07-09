@Author : ZhangPeiCheng
@updateTime: 2023.12.4  14:29:27
                                   ###The entire Thinking process of this idea
## summary:
    First version of predictor using image block MAE.
    => The preliminary results of the experiment now actually show that the key question is
    how to get the model to focus on << the pixels model needs to pay attention to >> when putting in a
    larger number of pixels?

## train set:
    division method: checkerboard cut masks

## Revises based on the results of the current experiment:
1.  checkerboard cut is not quiet suitable, because too much redundant information distort the results
    -> but I can not just set up with a new idea  (-_-||)
2.  Feature extraction needs to be performed manually to enhance the model performance, because without manual.
    feature, more encoders are needed for feature extraction, increasing the risk of over-fitting.
3.  The smaller the block selection, the better the results,which just prove that pixel selection
    method should be designed carefully.
4.  When we divide patch to pixel sequence,masked pixel is surrounded by two origin pixel,but we fail to take
    Top and bottom two pixels into consideration. A important factor of designing feature is that we ought to
    try to combine their representation into left and right pixels vector so that the features will be recognized
    by the model.
5.  try to randomize the divide method, using random sample to choose the pixel,but it is hard for this method to get feature
    (Wu`s job)


## Two ideas for feature extraction
1. Manually extract some spatial features, such as the variance operator in previous work,I can finish this job
2. Multiple convolutional kernels are added using a rich model-like idea, with specially designed convolutional
   kernels that allow the convolutional kernels to disregard information about masked pixels( Wu`s job )
   ->example: Focus on diagonal convolution kernel and inverse diagonal convolution kernel
   ->kernel:
   [[0, 0, 1]
    [0, 1, 0]
    [1, 0, 0]]

   [[1, 0, 0]
    [0, 1, 0]
    [0, 0, 1]]
    but now I just feel suspicious about this method cuz too much useless info will destroy the precision.
 ______________________________________________________________________________________________________________________
 !-|*_*|-!
 author`s psychological condition is worrying hhhhh. the result is still not good,But I have adapted the failure~~

 Some easy record during experiment:
 1. when patches size become larger, the depth of the encoder and decoder should be deeper, the hp should be adjusted carefully
 2. this another option I should try, that is.....is.... oh! I am fucking forget.Oh I get ,I should try to compute Loss between
    all real pixels of the patches and reconstructed pixels by decoder.
 3. different block size should use different block width and depth.

========================================================================================================================
{!}
 &_&]
 Brain Storming!
 A crucial modification method! (at least I think that is good)
 => 4. Modifying the decoder, you can think about partial masking of the pixel to be predicted, e.g.,
    I'm just masking off the pixel itself,but I can just add spatial operators to the masked area.
 example:                                                   _______
               encoder_output                              |decoder|
    [masking part] + [Spatial information]                 |       |
               encoder_output                      =>      |       |
    [masking part] + [Spatial information]                 |       |
               encoder_output                              |_______|
                     .
                     .
                     .
 god! I am a artist! hhhh
---> after experiment, this method is kind of useful,but there is still have just too mush useless image information
________________________________________________________________________________________________________________________
    1. I just suddenly figure out that with small block size and small parameters the effect seem to enhance,so when the degree of
pixel information congregate increase. the effect became better,which indicated that the scale of the model should be adjusted
with the input scale.(Forced stuffing the relevant pixels)
    2. Although the key pro is the attention mechanism always take too many pixels into consideration, why don not we just trying
to just allow a pixel only focus on some most relevant pixels? I suddenly find a mechanism called <<Sparse attention mechanism>>
maybe I should read some paper now,tired, have worked for almost 10 hours.... but I am interested, gosh...
________________________________________________________________________________________________________________________
    Several multiple variants of the attention mechanism:
->  Su`s code address: https://g0ithub.com/bojone/attention/blob/master/attention_keras.py
->  paper:<< Image Transformer >>
->  example code location: model/reference-model/several_attention_keras.py

————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    locality? just suddenly conv1d occur into my mind, I can use conv1d to extract some local feature?
why not just choice conv2d?(I can try this, but I don`t know the result)
    so: (Wu`s job)
    1. conv2d-> extract features
    2. manually extract features


