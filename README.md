# Face detection and emotion/gender classification using IMDB datasets with keras CNN models.

## Face detection

Leverages the algorithm proposed in *Deep Residual Learning for Image Recognition. In CVPR, 2016.*

## Gender and age estimation

Leverages the algorithm proposed in *DEX: Deep EXpectation of apparent age from a single image" in ICCV, 2015*

## Requirements

```
pip install keras
pip install tensorflow
pip install numpy
pip install h5py
pip install opencv2-python
```
## Usage

1. Detect faces from a mp4 video, and estimate age and gender for the detected face.

```
usage: video_detector.py [-h] [--video_input VIDEO_INPUT]
                         [--video_output VIDEO_OUTPUT]

optional arguments:
  -h, --help            show this help message and exit
  --video_input VIDEO_INPUT
  --video_output VIDEO_OUTPUT

```

2. Detect faces from an image, and estimate age and gender for the detected face.
```
usage: image_detector.py [-h] [--video_input VIDEO_INPUT]
                         [--video_output VIDEO_OUTPUT]

optional arguments:
  -h, --help            show this help message and exit
  --video_input VIDEO_INPUT
  --video_output VIDEO_OUTPUT
  --test
  --test2
```
