# UNET Training and Inference

This sub folder contains the training and inference scripts for a trained unet model for monitor segmentation.

Trained Model Paths:

Sno|Model | Epochs | IoU | Encoder depth | Pretrained | Backbone | Link
--|--|--|--|--|--|--|--|
1| UNet |4|0.9091|5|No|Resnet34|[Link 1](https://drive.google.com/file/d/1BhWIpEVEQlhlFGIlFQmO2Qrw3R7WeokM/view?usp=sharing)
2| UNet|10|0.8888|3|No|Resnet34|[Link 2](https://drive.google.com/file/d/1sVMY0qMtR9xgsFf32j8MN96pxOwyO__b/view?usp=sharing)
2| UNet|10|0.9097|5|Yes|Resnet34|[Link 2](https://drive.google.com/file/d/1eK6s2vVWTVQfUMWFiGMwpAslU8IyG_HD/view?usp=sharing)


Run the following command to inference on an image folder:
```
python test.py --model-path model.pth\
 --image-folder ./image_folder\
  --output-dir ./output_folder\
```


**Training script is untested. Use Kaggle to train.**
To use on colab run:

```
!pip install segmentation-models-pytorch
```

