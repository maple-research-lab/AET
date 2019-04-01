# AET vs. AED: Unsupervised Representation Learning by Auto-Encoding Transformations rather than Data
The project website for "Auto-Encoding Transformations."

## Abstract 
The success of deep neural networks often relies on a large amount of labeled examples, which can be difficult to obtain in many real scenarios. To address this challenge, unsupervised methods are strongly preferred for training neural networks without using any labeled data. In this paper, we present a novel paradigm of unsupervised representation learning by Auto-Encoding Transformation (AET) in contrast to the conventional Auto-Encoding Data (AED) approach. Given a randomly sampled transformation, AET seeks to predict it merely from the encoded features as accurately as possible at the output end. The idea is the following: as long as the unsupervised features successfully encode the essential information about the visual structures of original and transformed images, the transformation can be well predicted. We will show that this AET paradigm allows us to instantiate a large variety of transformations, from parameterized, to non-parameterized and GAN-induced ones. Our experiments show that AET greatly improves over existing unsupervised approaches, setting new state-of-the-art performances being greatly closer to the upper bounds by their fully supervised counterparts on CIFAR-10, ImageNet and Places datasets.

## Formulation

| ![AED](https://github.com/maple-research-lab/AET/blob/master/resource/AED.png) |
|:--:| 
| *(a) Auto-Encoding Data* |
| ![AET](https://github.com/maple-research-lab/AET/blob/master/resource/AET.png) |
| *(b) Auto-Encoding Transformation* |
| *Figure 1. An illustration of the comparison betweeen AED and AET models. AET attempts to estimate the input transformation rather than the data at the output end. This forces the encoder network E to extract the features that contain the sufficient information about visual structures to decode the input transformation.* |

Figure 1 illustrates our idea of auto-encoding transformation (AET) in comparison with the conventional auto-encoding data (AED). We build a transformation decoder D to reconstruct the input transformation t from the representations of an original image E(x) and the transformed image E(t(x)), where E is the representation encoder. 

The least-square difference between the estimated transformation and the original transformation is minimized to train D and E jointly. For details, please refer to [our paper](https://arxiv.org/abs/1901.04596).

## Run our codes

### Cifar10
    cd cifar/affine
or

    cd cifar/projective
Unsupervised Learning:
    CUDA_VISIBLE_DEVICES=0 python main.py --cuda --outf ./output --dataroot $YOUR_CIFAR10_PATH$ 

Supervised Evaluation with two FC layers:
    python classification.py --dataroot $YOUR_CIFAR10_PATH$ --epochs 200 --schedule 100 150 --gamma 0.1 -c ./output_cls --net ./output/net_epoch_1499.pth --gpu-id 0

### ImageNet 

## Citation

Liheng Zhang, Guo-Jun Qi, Liqiang Wang, Jiebo Luo. AET vs. AED: Unsupervised Representation Learning by Auto-Encoding Transformations rather than Data,  in Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2019), Long Beach, CA, June 16th - June 20th, 2019. [[pdf](https://arxiv.org/abs/1901.04596)]

## Disclaimer

Some of our codes reuse the github project [FeatureLearningRotNet](https://github.com/gidariss/FeatureLearningRotNet).  

## License

This code is released under the MIT License.



