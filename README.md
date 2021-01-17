# Fast-Style-Transfer
A PyTorch implementation of the paper "Perceptual Losses for Real-Time Style Transfer and Super-Resolution" by Justin Johnson, Alexandre Alahi, and Li Fei-Fei. This is just my adaptation of their work in PyTorch. Thank you also to [this helpful repo](https://github.com/gordicaleksa/pytorch-neural-style-transfer-fast) from gordicaleksa.

If you would like to download the trained weights to run style transfer on your own images, [click here](https://www.dropbox.com/sh/krumw3lpyn89tgj/AACEqkc8I-7_fzF089cfiwnYa?dl=0).

## Feed Forward Neural Style Transfer in PyTorch
Neural Style Transfer is one of those awesome things in deep learning that is not too difficult to understand, not too difficult to get started with, and produces great results. The traditional method for NST actually trains the pixels of an image to jointly minimize a style loss for one image and a content loss for a different image (check out the original paper [here](https://arxiv.org/abs/1508.06576)). This gets great results, but it has to "train" each image, so it can take a while, especially if you don't have access to GPUs. Fortunately, Johnson et al. showed in [this paper](https://arxiv.org/abs/1603.08155) that it is possible to train a convolutional neural network to do this style transfer. Training, then, takes a long time, but the actual style transfer is quick using the trained weights.

This repo is intended to be a simple implementation of that Johnson et al. paper in PyTorch. There are other repositories on GitHub which already do this, but I created this repo to be as "beginner friendly" as possible. There are no custom wrappers for common PyTorch layers, classes are as simple and clean as they can be with readability favored over the shortest possible code, and comments and docstrings are included to be as descriptive as possible. The structure of the repo is designed to be simple as well: the file 'train.py' handles all training of new models and has command line arguments to fine tune that training while the file 'run.py' is what actually executes the style transfer using trained weights. Everything is as simple as can be and I hope this is helpful to someone.

### My Results
I trained this network on 7 different style images and overall I am pretty happy with the results. One thing I noticed (and this may be helpful if you want to train on your own style images) is that certain style images will look way better than others as a final product. You can see for yourself in my results comparison below, but notice for instance that the Monet output does not really look very good. It is capturing the style of that Monet painting with the bridge over water lillies, but that style doesn't really fit with the very detailed skyline of Seattle and the horizontal lines of the bridge seem to be so strong that the network learns to place horizontal lines throughout the Seattle image, which is a very unattractive look. Similarly, the Okeefe image has so little detail that it overly smooths the complex Seattle skyline. I tested that image on one of Torres del Paine as well and it looked better, but it would be best suited for a very simple scene. Anyways, I guess my point is that the results vary greatly depending on what style image you choose.

![my results](https://github.com/rileypsmith/Fast-Style-Transfer/blob/main/results/Result%20Comparison.png)

### Running trained models on your own images
To run one of the trained models on your own images, first download the weight file(s) [here](https://www.dropbox.com/sh/krumw3lpyn89tgj/AACEqkc8I-7_fzF089cfiwnYa?dl=0). Put them inside a folder in the main project folder called 'trained_weights'. For instance, if you want to run the Cezanne weights, you should have them at the path "Fast-Style-Transfer/trained_weights/Cezanne.pth". Do not change the name of the weights file (this is important). Once it's in that folder, simply run from the command line:
```
cd Fast-Style-Transfer
python run.py --artist Cezanne --content_image <path_to_your_image> --output_file <optional_specific_output_path>
```
You don't need `cd Fast-Style-Transfer` if you are already in that folder. Just cd into the project folder and run that second command. If you do not specify an output path, the output image will be written to "results/Cezanne.png" in this case.

### Training your own models
To train on your own style image, find one you like and put it in the folder "data/style". I recommend renaming it to "<artist_name>.jpg". You will also need to have a dataset to train on. I used the MS Coco dataset and I recommend you do the same. You can download the dataset [here](https://cocodataset.org/#download) and you should put it inside a folder called "data/coco" in the main project folder. By default, this is the directory that will be used for training images. If you would like to train on a different dataset, simply use the `--train_dir <your_training_directory>` flag when you run the training script.

With your style image selected and dataset downloaded, run
```
python train.py --style_image data/style/<artist_name>.jpg
```
That will start training! There are lots of other command line options. A few I will note and my recommended settings for them:
```
--batch_size 4 --image_checkpoint 100 --save_checkpoint 500
```
These are what I used during most of my training. `--image_checkpoint` is how often to save a progress image (to see how the network is doing visually). `--save_checkpoint` is how often (how many batches) to save the network's weights. So those are important. Also `--batch_size 4` is optimized for CPU. If you have one or more GPUs, you may be able to use a larger batch size. 

Definitely play around with the content and style weight as well (using the `--content_weight` and `--style_weight` flags). The defaults are what I found worked most of the time but different style images required different parameters so play around.

### Helpful tips
Definitely use the image checkpoints. They will automatically be saved to the "logs/images" folder and they are really helpful to see how the network is doing. If you notice midway through training that the network has diverged (like all of a sudden one of the progress images is way off), it usually means your STYLE LOSS IS TOO HIGH! This was a common problem for me.

If that happens, you can stop the network, reduce the style loss, and resume training. To do so, stop the training and then run
```
python train.py --resume weights/EPOCH_<epoch>_BATCH_<batch>.pth
```
But of course use the epoch and batch number you want to resume from. Those weights will be automatically stored during training (unless you disable that which I do not recommend). Using that command will pick up training where you left off. Just be sure to re-specify the batch size and of course adjust the style weight.

### Contributing
If you have features you'd like to see, submit a feature request! I won't be maintaining this full time but I think it would be fun to build this out into a versatile and really well-rounded NST repo with the help of others so if there's a feature you want let me know! If you make a feature yourself and you think others might like it too, submit a pull request! I'm happy to review it and build this thing out.
