# ColorNet
This is a repo to help support the paper entitled *Enabling automated herbarium sheet image post-processing through neural network models for color reference chart detection*. 

## Modified Faster R-CNN
A modified Keras implementation of Faster R-CNN for Herbaria CRC detection is included. The original Keras implementation, including its training/testing instructions may be found here: https://github.com/you359/Keras-FasterRCNN

## Trained models:
You may find all of the trained models within this link: https://drive.google.com/drive/folders/14W3oqSEb1bpzsHk4GB3ABAwFZSY91IIH?usp=sharing

## Current model versions:
- Small CRC
  - Region Proposal: version 17k, "mlp_histogram_v17k.hdf5"
  - Classifier: version 6c, "discriminator-v6c.tflite"
- Large CRC
  - Modified Faster R-CNN: "model_frcnn.hdf5"
