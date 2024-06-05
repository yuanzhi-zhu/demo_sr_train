## A lite version of [KAIR](https://github.com/cszn/KAIR/) for SISR

- Download [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) datasets.
- put the datasets in `datasets/DIV2K` and `datasets/Flickr2K`, respectively.
- modify the dataroot_H in option file correspondingly
- python main_train_psnr.py --opt options/train_msrresnet_psnr.json

