# PawPal
Entertaining and Training your dog while you are away.

Using state-of-the-art computer vision algorithms, this dog localization and activity recognition system can determine what your dog is doing from a home surveillance camera.

## Requirements
Python3, Tensorflow, Numpy, OpenCV

[Darkflow](https://github.com/thtrieu/darkflow) and Yolo

[M-PACT](https://github.com/MichiganCOG/M-PACT) Activity Recognition Platform


## Results

![](https://github.com/ehofesmann/PawPal/blob/master/images/biting.png)
![](https://github.com/ehofesmann/PawPal/blob/master/images/furniture.png)



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
Install M-PACT and copy c3d_frozen into the models directory of M-PACT






