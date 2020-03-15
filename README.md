# cnn_style_transfer

My PyTorch implementation of [Perceptual Losses for Real-Time Style Transfer
and Super-Resolution](https://arxiv.org/pdf/1603.08155.pdf)

Johnson et al. use two networks for their style transfer model:
* A *loss network* (VGG-16), which is pre-trained for image classification and defines the perceptual loss that measures perceptual differences in content and style between images. This network is fixed during training.
* An *image transformation network*, which is an auto-encoder that transform an input image into a stylized output image.
Network exact architecture can be found [https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf](here).

#### Important quotes:

For style transfer the input and output are both color images of shape 3 × 256 × 256 (during training).

Since the image transformation networks are fully-convolutional, at test-time they can be applied to images of any resolution.