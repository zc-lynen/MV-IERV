# Implicit-explicit Integrated Representations for Multi-view Video Compression

## Get started
Run with Python 3.7, and set up a conda environment as follows:
```
pip install -r requirements.txt 
```

## High-Level structure
The code is organized as follows:
* [main.py](./main.py) includes a generic training and testing routine.
* [IERV.py](./model/IERV.py) includes the implicit network structure.
* [--dataset] directory of image dataset of all views
* [--dataset_basrec] directory of basic view reconstruction
* log files will be saved in output directory (specified by ```--outf```)


## Preprocessing
Use `ffmpeg` to convert multi-view video to images.
```
ffmpeg -s {width}x{height} -pix_fmt xx -i view0_yuv_path dataset/f%05d_v00.png

ffmpeg -s {width}x{height} -pix_fmt xx -i view1_yuv_path dataset/f%05d_v01.png

...
```

Use x265 (or other 2D codec) to encode the basic view (view index: b_idx), and then use `ffmpeg` to convert the reconstructed view into images.
```
ffmpeg -s {width}x{height} -i basic_yuv_path -c:v hevc -preset veryfast -crf {crf} bin_path

ffmpeg -i bin_path basic_yuv_rec_path

ffmpeg -s {width}x{height} -pix_fmt xx -i basic_yuv_rec_path dataset_basrec/f%05d_v{%02d, b_idx}.png
```

## Reproducing experiments

## Model Representation

#### Training experiments
The experiment can be reproduced with ```9_16_20```, ```9_16_40``` and ```9_16_80``` for ```fc_hw_dim``` respectively.
```
python main.py -e 300 --dataset xx --dataset_basrec xx --outf xx \
    --fea_hw_dim 9_16_20 --expansion 1 --lower_width 96 --strides 5 3 2 2 2 \
    --reduction 2 --norm in --act gelu \
    --view_num xx --frame_perview xx --basic_ind xx --resol 1920 1080 \
    --warmup 0.2 -b 1 --lr 0.0005 --qbit -1
```

#### Evaluation experiments
To evaluate pre-trained model, just add --eval_Only and specify model path with --weight, you can specify model quantization with ```--quant_bit [bit_lenght]```, yuo can test decoding speed with ```--eval_fps```, below we preovide sample commends for NeRV-S on bunny dataset
```
python main.py -e 300 --dataset xx --dataset_basrec xx --outf xx \
    --fea_hw_dim 9_16_20 --expansion 1 --lower_width 96 --strides 5 3 2 2 2 \
    --reduction 2 --norm in --act gelu \
    --view_num xx --frame_perview xx --basic_ind xx --resol 1920 1080 \
    --warmup 0.2 -b 1 --lr 0.0005 --qbit -1 \
    --weight xx --eval_only
```

#### Decoding: Dump predictions with pre-trained model 
To dump predictions with pre-trained model, just add ```--dump_images``` besides ```--eval_Only``` and ```--weight```
```
python main.py -e 300 --dataset xx --dataset_basrec xx --outf xx \
    --fea_hw_dim 9_16_20 --expansion 1 --lower_width 96 --strides 5 3 2 2 2 \
    --reduction 2 --norm in --act gelu \
    --view_num xx --frame_perview xx --basic_ind xx --resol 1920 1080 \
    --warmup 0.2 -b 1 --lr 0.0005 --qbit -1 \
    --weight xx --eval_only --dump_images
```

## Model Compression

Coming soon.

## Acknowledgement

This repository is based on [NeRV](https://github.com/haochen-rye/NeRV).
