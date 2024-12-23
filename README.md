# PLMAE
Article Link:https://doi.org/10.1016/j.sigpro.2024.109664
## The Structure of The Model

![the structure of the PLMAE](https://github.com/cheng-zhangpei/PLMAE/blob/7c294174378c31896460d65409028b4d8c1dccc2/image/revision.png)
## Model Training
- **Entry Point**: The model training process is initiated through the `__init__training_0.75.py` file.

## Model Prediction

- **Entry Point**: The model prediction can be executed via the `model_predict.py` file located under the `interface` package.

## Model Location

- **Model Files**: The models are located under the `model/predict/predict_v2_conv` directory.

## trained Models

- **trained Model**: The trained models we use can be found in the `trained_model` directory.

## Using Model for RDHEI
- **RDHEI**: The overall method can be found in the `PLMAE_RDHEI_CODE` directory'.the code is writen by matlab.
- **introduction**:The overall processing of the RDHEI method for this code is as follows:
    First, use the PLMAE model to input a masked image and obtain a predicted image (since the prediction uses retained pixels, which are independent of carrier pixels, the generation of the predicted image can be independent of the dataembedding). As the output of the predicted image is in decimal form, the prediction values are output as txt files rather than image format. The code comes with six experimental images and their corresponding predicted images.
    Then, place the original image and the txt-formatted predicted image in the folder where the code resides. In the main code file RDHEI.m, change the experimental parameters (file name, codeword length N, and message bits K), and you can run the code to test the experimental results. Note that since this method is the VRAE RDHEI method, its maximum capacity depends on the code rate K/N when fully reversibility can be achieved. Therefore, after fixing the value of N, the value of K needs to be adjusted continuously, observing the code output until a stable fully reversible result is obtained.
