# P&sup2;-GAN Fast Style Transfer

Code and pre-trained models for "[P&sup2;-GAN: Efficient Style Transfer Using Single Style Image](https://arxiv.org/abs/2001.07466)".

![front page image](https://github.com/i-evi/p2gan/raw/master/resources/front.png)

- [P&sup2;-GAN Fast Style Transfer](#P&sup2;-GAN-Fast-Style-Transfer)
    - [Dependence](#dependence)
    - [Training](#training)
    - [Advanced Training Configurations](#advanced-training-configurations)
    - [Testing](#Testing)
    - [Experimental Configurations for Testing](#experimental-configurations-for-testing)
    - [Pre-trained models](#pre-trained-models)


## Dependence

* opencv-python
* tensorflow 1.x

This project was implemented in tensorflow, and used `slim` API, which was removed in tensorflow 2.x, thus you need running it on tensorflow 1.x.

## Training

**Dataset**
It doesn't need a very large dataset, Pascal VOC Dataset is good enough. For a dataset like VOC2007, it contains about 10K images, training 2~3 epochs can gain a good result.

**Pre-trained VGG Model**
The VGG16 model we used is here: [tensorflow-vgg](https://github.com/machrisaa/tensorflow-vgg), you need to download the weights file before training.

**Training Command**
After you prepared the pre-trained VGG model and dataset, you can train a model by the command as follows:

```bash
python train.py --model model_path --style style_image_path --dataset dataset_path

```
* `--model`: path to save the trained model.
* `--style`: path to the style image.
* `--dataset`: path to dataset.

For example:

```bash
python train.py --model model_save --style style/Van-Gogh-The-Starry-Night.jpg --dataset /home/username/Downloads/VOCdevkit/VOC2007/JPEGImages/ --lambda 5e-6
```
Further, we added an argument `--lambda` in this example, it's the hyper parameter between 1e-6~1e-7 to balance content and style.

**Training Control**
If you want to change the optimizer configuration, you need to edit `train.py`.

In each iteration while the model is training, `cfg.json` will be reloaded, thus some configuration can be set on training time:
* `epoch_lim`: how many training epochs will take.
* `preview`: allow to render preview images while training.
* `view_iter`: if `preview` valued `true`, render preview images at that iteration.

## Advanced Training Configurations

**Custom Patch Size**
Sometimes the texture primitives may be larger than the default patch size, which is set to 9. You can use a larger patch size by the option `--ps`. In this demo version code, supported patch size configurations are `9, 12, 15` and `16`.

For example:

```bash
python train.py --ps 15 --model model_save --style style/Van-Gogh-The-Starry-Night.jpg --dataset /home/username/Downloads/VOCdevkit/VOC2007/JPEGImages/ --lambda 5e-6
```

Training with the command above, the patch size will be set to 15 * 15.

**Custom Generator Structure**
This customization won't be supported under a command line option, it's experimental. Since the larger patch size will introduce more complex texture, and the generator is aiming at a lightweight network, there is only one residual block by the default configuration. Edit `model.py`, in the `g_residual_cfg`, set `l_num` to the number of residual blocks you want, for example `3` for 3 residual blocks.

## Testing

Choose a model and run the command to test the model:

```bash
python render.py --model model_path --inp input_path --oup output_path --size number [--cpu true]
```
* `--model`: Choose a mode.
* `--inp`: Path to input images.
* `--onp`: Path to save synthetic images.
* `--size`: Processing image size.
* `--cpu`: Optional, set `true` will processing by CPU.

For example:

```bash
python render.py --model model_save --inp /home/username/Pictures/Wallpaper/ --oup output --size 256
```

Note that `--inp` must be a directory, and `render.py` will process all images(extension with `jpg`, `bmp`, `png` and `jpeg`) under the directory. And the output directory must exist.

## Experimental Configurations for Testing

**Noise Control**
Sometimes, adding noise on the input images can bring good visual effect. The input images won't be added in any noise by default, we implemented a Gaussian noise control option `--noise`, for example:

```bash
python render.py --model model_save --inp /home/username/Pictures/Wallpaper/ --oup output --size 512 --noise 0.1
```

**Aspect Ratio**
In `render.py`, the input tensor shape will be configured while the tensorflow graph be initialized. The height and width of the input tensor are equal and set by the option `--size`. Therefore, the processing of the input images may not keep the aspect ratio, thus cause texture distortion. `render-keep-ratio.py` can keep the aspect ratio of the input image resize, but process only one input image each time. The command line usages are compatible with `render.py`, except `--inp`, you should specify an image file here. For example:

```bash
python render-keep-ratio.py --model model_save --inp /home/username/Pictures/Wallpaper/test.jpg --oup output --size 1024
```

## Pre-trained models
We have trained some models, they are located at `~/available_models/`:

| Style image | Content | Stylized |Content | Stylized | Model name |
|-------------|---------|----------|--------|----------|------------|
|![](https://github.com/i-evi/p2gan/raw/master/resources/style/Mosaic.jpg)|![](https://github.com/i-evi/p2gan/raw/master/content/9.jpg)|![](https://github.com/i-evi/p2gan/raw/master/resources/demo/5_0.jpg)|![](https://github.com/i-evi/p2gan/raw/master/content/1.jpg)|![](https://github.com/i-evi/p2gan/raw/master/resources/demo/5_1.jpg)|Mosaic|
|![](https://github.com/i-evi/p2gan/raw/master/resources/style/Matisse-Woman-with-a-Hat.jpg)|![](https://github.com/i-evi/p2gan/raw/master/content/6.jpg)|![](https://github.com/i-evi/p2gan/raw/master/resources/demo/3_0.jpg)|![](https://github.com/i-evi/p2gan/raw/master/content/7.jpg)|![](https://github.com/i-evi/p2gan/raw/master/resources/demo/3_1.jpg)|Matisse-Woman-with-a-Hat|
|![](https://github.com/i-evi/p2gan/raw/master/resources/style/Rough-Sea-1917-woodcut.jpg)|![](https://github.com/i-evi/p2gan/raw/master/content/14.jpg)|![](https://github.com/i-evi/p2gan/raw/master/resources/demo/8_0.jpg)|![](https://github.com/i-evi/p2gan/raw/master/content/15.jpg)|![](https://github.com/i-evi/p2gan/raw/master/resources/demo/8_1.jpg)|Rough-Sea-1917-woodcut|
|![](https://github.com/i-evi/p2gan/raw/master/resources/style/Qi-Baishi-FengYeHanChan.jpg)|![](https://github.com/i-evi/p2gan/raw/master/content/12.jpg)|![](https://github.com/i-evi/p2gan/raw/master/resources/demo/7_0.jpg)|![](https://github.com/i-evi/p2gan/raw/master/content/13.jpg)|![](https://github.com/i-evi/p2gan/raw/master/resources/demo/7_1.jpg)|Qi-Baishi-FengYeHanChan|
|![](https://github.com/i-evi/p2gan/raw/master/resources/style/Matisse-Landscape-at-Collioure.jpg)|![](https://github.com/i-evi/p2gan/raw/master/content/10.jpg)|![](https://github.com/i-evi/p2gan/raw/master/resources/demo/6_0.jpg)|![](https://github.com/i-evi/p2gan/raw/master/content/11.jpg)|![](https://github.com/i-evi/p2gan/raw/master/resources/demo/6_1.jpg)|Matisse-Landscape-at-Collioure|
|![](https://github.com/i-evi/p2gan/raw/master/resources/style/Jay-DeFeo-Mountain-No.2.jpg)|![](https://github.com/i-evi/p2gan/raw/master/content/8.jpg)|![](https://github.com/i-evi/p2gan/raw/master/resources/demo/4_0.jpg)|![](https://github.com/i-evi/p2gan/raw/master/content/9.jpg)|![](https://github.com/i-evi/p2gan/raw/master/resources/demo/4_1.jpg)|Jay-DeFeo-Mountain-No.2|
|![](https://github.com/i-evi/p2gan/raw/master/resources/style/Jackson-Pollock-Number-14-Gray-1948.jpg)|![](https://github.com/i-evi/p2gan/raw/master/content/0.jpg)|![](https://github.com/i-evi/p2gan/raw/master/resources/demo/9_0.jpg)|![](https://github.com/i-evi/p2gan/raw/master/content/16.jpg)|![](https://github.com/i-evi/p2gan/raw/master/resources/demo/9_1.jpg)|Jackson-Pollock-Number-14-Gray-1948|
|![](https://github.com/i-evi/p2gan/raw/master/resources/style/NASA-Universe.jpg)|![](https://github.com/i-evi/p2gan/raw/master/content/4.jpg)|![](https://github.com/i-evi/p2gan/raw/master/resources/demo/2_1.jpg)|![](https://github.com/i-evi/p2gan/raw/master/content/5.jpg)|![](https://github.com/i-evi/p2gan/raw/master/resources/demo/2_0.jpg)|NASA-Universe|
|![](https://github.com/i-evi/p2gan/raw/master/resources/style/Leonardo-da-Vinci-Helicopter.jpg)|![](https://github.com/i-evi/p2gan/raw/master/content/1.jpg)|![](https://github.com/i-evi/p2gan/raw/master/resources/demo/0_1.jpg)|![](https://github.com/i-evi/p2gan/raw/master/content/0.jpg)|![](https://github.com/i-evi/p2gan/raw/master/resources/demo/0_0.jpg)|Leonardo-da-Vinci-Helicopter|
|![](https://github.com/i-evi/p2gan/raw/master/resources/style/Van-Gogh-The-Starry-Night.jpg)|![](https://github.com/i-evi/p2gan/raw/master/content/3.jpg)|![](https://github.com/i-evi/p2gan/raw/master/resources/demo/1_1.jpg)|![](https://github.com/i-evi/p2gan/raw/master/content/2.jpg)|![](https://github.com/i-evi/p2gan/raw/master/resources/demo/1_0.jpg)|Van-Gogh-The-Starry-Night|
|![](https://github.com/i-evi/p2gan/raw/master/resources/style/Robert-Delaunay-Portrait-de-Metzinger.jpg)|![](https://github.com/i-evi/p2gan/raw/master/content/16.jpg)|![](https://github.com/i-evi/p2gan/raw/master/resources/demo/b_0.jpg)|![](https://github.com/i-evi/p2gan/raw/master/content/17.jpg)|![](https://github.com/i-evi/p2gan/raw/master/resources/demo/b_1.jpg)|Portrait-de-Metzinger(full)|
|![](https://github.com/i-evi/p2gan/raw/master/resources/style/Robert-Delaunay-Portrait-de-Metzinger-part.jpg)|![](https://github.com/i-evi/p2gan/raw/master/content/16.jpg)|![](https://github.com/i-evi/p2gan/raw/master/resources/demo/a_0.jpg)|![](https://github.com/i-evi/p2gan/raw/master/content/17.jpg)|![](https://github.com/i-evi/p2gan/raw/master/resources/demo/a_1.jpg)|Portrait-de-Metzinger(part)|

