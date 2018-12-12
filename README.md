# PawPal
Entertaining and Training your dog while you are away.

Using state-of-the-art computer vision algorithms, this dog localization and activity recognition system can determine what your dog is doing from a home surveillance camera.

![](https://github.com/ehofesmann/PawPal/blob/master/images/pipeline.png)


## Requirements
Python3, Tensorflow, Numpy, OpenCV

[Darkflow](https://github.com/thtrieu/darkflow) for Yolo

[M-PACT](https://github.com/MichiganCOG/M-PACT) Activity Recognition Platform


## Results
![](https://github.com/ehofesmann/PawPal/blob/master/images/furniture.png)

#### Biting C3D Recogntiion Accuracy (%)
|  Mean | Standard Deviation | Random Chance |  
|:----------:|:------:| :----:|
|  68.41 | 6.10 | 50.00 |

Across 5 splits given in ```tfrecords_pawpal/split.npy``` in the dataset download link below


![](https://github.com/ehofesmann/PawPal/blob/master/images/biting.png)



## Usage

### Setup
Download the weights for C3D [Download link](https://umich.box.com/s/va0jkzx6ym0vb4k6909sxebjijne0uez)

Add the weight file to ```PawPal/c3d/```.

### Testing

```
python detection_c3d.py --vidnum 0
```

### Training

#### Dataset
Dog biting vs non biting tfrecords dataset [Download link](https://umich.box.com/s/jptvbcuig2ieejmhhv7p8kic7t3vraeu)

![](https://github.com/ehofesmann/PawPal/blob/master/images/data.png)


#### Activity Recognition Model
Install M-PACT and copy ```PawPal/c3d/c3d_frozen``` into the models directory of M-PACT






