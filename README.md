# HTC-Net
The codes for the work "HTC-Net: 

## 1. Download pre-trained swin transformer model (Swin-T)
* [Get pre-trained model in this link] (https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY?usp=sharing): Put pretrained Swin-T into folder "pretrained_ckpt/" and create dir 'chechpoint','test_log' in the root path.

## 2. Prepare data

- You can go to https://challenge.isic-archive.com/data/#2017 to acquire the ISIC-2017 dataset.
- You can go to https://paperswithcode.com/dataset/kvasir to acquire the Kvasir-SEG dataset.

## 3. Environment

- Please prepare an environment with python=3.8, and then use the command "pip install -r requirements.txt" for the dependencies.

## 4. Train/Test

- Run the train script on the ISIC-2017 and the COVID-QU-Ex dataset. The batch size we used is 8. If you do not have enough GPU memory, the bacth size can be reduced to 4 or 6 to save memory. For more information, contact 2639442956@qq.com.

- Train

```
python train.py 
```
- Test 

```
python test.py 
```

## References
* [TransUnet](https://github.com/Beckschen/TransUNet)
* [SwinTransformer](https://github.com/microsoft/Swin-Transformer)
* [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet.)
