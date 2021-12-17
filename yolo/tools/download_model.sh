#!/bin/bash

VERSION=$1


if [ "$VERSION" == "yolov3" ]
then
    wget https://pjreddie.com/media/files/yolov3.weights -O model/"$VERSION".weights
elif [ "$VERSION" == "yolov4" ]
then
    wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights -O model/"$VERSION".weights
else
    echo Currently only yolov3 and v4 are supported.
fi
