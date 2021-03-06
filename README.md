# cnn_style_transfer

My PyTorch implementation of [Perceptual Losses for Real-Time Style Transfer
and Super-Resolution](https://arxiv.org/pdf/1603.08155.pdf)

Johnson et al. use two networks for their style transfer model:
* A *loss network* (VGG-16), which is pre-trained for image classification and defines the perceptual loss that measures perceptual differences in content and style between images. This network is fixed during training (all requires_grad are changed to False).
* An *image transformation network*, which is an auto-encoder that transform an input image into a stylized output image.
Network exact architecture can be found [here](https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf).

#### Usage

All the following commands must be executed from above the 'src' folder.

To train the model use the following command:
```console
$ python src/train.py
```
To understand all the possible arguments, type:
```console
$ python src/train.py -h
```

To stylize an image with a trained model use the following command:
```console
$ python src/stylize.py trained_transfer_net_filename image_to_stylize_filename
```
Once again, to understand all the possible arguments, type:
```console
$ python src/stylize.py -h
```

#### Informations:

*For style transfer the input and output are both color images of shape 3 × 256 × 256 (during training).*

*Since the image transformation networks are fully-convolutional, at test-time they can be applied to images of any resolution.*

*\[The last layer\] uses a scaled tanh to ensure that the output image has pixels in the range \[0, 255\].* I changed this by a sigmoid to scale the output between 0 and 1.

Instance Normalization gives better results than a simple batch norm: [Instance Normalization](https://arxiv.org/pdf/1607.08022.pdf)

I use reflection padding since it gives cleaner results.

I use an upsampling followed by a conv2d instead of a convTranspose2d, which should [give results with less artifacts](https://distill.pub/2016/deconv-checkerboard/).

Notebooks in the src folder are used to test that all functionalities work as intended and the *preprocess* notebook is used to delete the grayscaled images from the dataset.