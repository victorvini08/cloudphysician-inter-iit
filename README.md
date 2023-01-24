# Main Pipeline

## Segmentation

The code for performing segmentation is present inside `segment.py`. 
- First instantiate the class `Segment` using the `config.yaml` file. 
- Make sure the `unet-model-path` is set inside the `config.yaml` file appropriately.
- Call the method `run` which takes as input an image file of the health monitor which should be a numpy array of dimentions `(H,W,3)`.
- If Python Gods are with you, the segmented screen should be returned as a numpy array of dimensions `(H,W,3)`.