

##################################
 README IS BEST VIEWED ON GITHUB:
https://github.com/ehofesmann/PawPal/

##################################






# PawPal
Entertaining and Training your dog while you are away.

Using state-of-the-art computer vision algorithms, this dog localization and activity recognition system can determine what your dog is doing from a home surveillance camera.

![](https://github.com/ehofesmann/PawPal/blob/master/images/pipeline.png)



## Results
![](https://github.com/ehofesmann/PawPal/blob/master/images/furniture.png)

#### Biting C3D Recogntiion Accuracy (%)
|  Mean | Standard Deviation | Random Chance |  
|:----------:|:------:| :----:|
|  68.41 | 6.10 | 50.00 |

Across 5 splits given in ```tfrecords_pawpal/split.npy``` in the dataset download link below


![](https://github.com/ehofesmann/PawPal/blob/master/images/biting.png)



## Usage



### Requirements
Python 3.5

OpenCV 

Tensorflow 1.0.0

Cython


[Darkflow](https://github.com/thtrieu/darkflow) for Yolo

[M-PACT](https://github.com/MichiganCOG/M-PACT) Activity Recognition Platform

Detailed installation instructions below.

### Installation and Setup

Follow instructions below to install darkflow and PawPal
```
git clone https://github.com/thtrieu/darkflow
virtualenv -p python3.5 env
source env/bin/activate
pip install tensorflow==1.0.0 
pip install Cython 
pip install opencv-python
cd darkflow
sudo apt-get install python3 python-dev python3-dev \
     build-essential libssl-dev libffi-dev \
     libxml2-dev libxslt1-dev zlib1g-dev \
     python-pip
pip install -e .
flow  (ignore any errors)
cd ..
git clone https://github.com/ehofesmann/PawPal/
cd PawPal
mkdir ../weights/

```
Download the weights for C3D [Download link](https://umich.box.com/s/va0jkzx6ym0vb4k6909sxebjijne0uez)

Download yolo.weights [Download link](https://drive.google.com/drive/folders/0B1tW_VtY7onidEwyQ2FtQVplWEU)
```
mv ~/Downloads/checkpoint-532.npy ../weights/
mv ~/Downloads/yolo.weights ../darkflow/bin/
```
Update the /path/to/darkflow in detect_video.py
```
python detect_video.py --vidpath example/example1.mp4
```

### Testing

```
python detect_video.py --vidpath example/example1.mp4
```

### Training or Finetuneing

#### Dataset
Dog biting vs non biting tfrecords dataset [Download link](https://umich.box.com/s/jptvbcuig2ieejmhhv7p8kic7t3vraeu)

![](https://github.com/ehofesmann/PawPal/blob/master/images/data.png)


#### Activity Recognition Model
Install M-PACT and copy ```PawPal/c3d/c3d_frozen``` into the models directory of M-PACT






