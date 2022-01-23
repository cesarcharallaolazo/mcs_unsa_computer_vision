# mcs_unsa_computer_vision
Repository for Computer Vision course from MCS UNSA

## Training

1. Spin up an amazon aws ec2 instance with gpu
2. Configura Yolov3 with Darknet53 backbone
3. Run
    
    ./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg backup/yolov3-voc.backup
    
## Prediction

We test Yolov3 with Python 3.8 scripts and these libraries:
- certifi==2021.10.8
- distro==1.6.0
- numpy==1.22.0
- opencv-python==3.4.8.29
- packaging==21.3
- pyparsing==3.0.6
- scikit-build==0.12.0

1. Test Yolov3 over images: python trabajo_final/detect_object.py
2. Test Yolov3 over videos: python trabajo_final/detect_object_video.py

