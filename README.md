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
* `--dataset` directory of image dataset of all views
* `--dataset_basrec` directory of basic view reconstruction
* log files will be saved in output directory (specified by ```--outf```)


## Preprocessing
Use `ffmpeg` to convert multi-view video to images.
```
ffmpeg -s {width}x{height} -pix_fmt xx -i view0_yuv_path dataset/f%05d_v00.png

ffmpeg -s {width}x{height} -pix_fmt xx -i view1_yuv_path dataset/f%05d_v01.png

...
```

Use `x265` (or other 2D codec) to encode the basic view (view index of `b_idx`), and then use `ffmpeg` to convert the reconstructed view into images.
```
ffmpeg -s {width}x{height} -i basic_yuv_path -c:v hevc -preset veryfast -crf {crf} bin_path

ffmpeg -i bin_path basic_yuv_rec_path

ffmpeg -s {width}x{height} -pix_fmt xx -i basic_yuv_rec_path dataset_basrec/f%05d_v{%02d, b_idx}.png
```

## Model Representation Experiments

### Training experiments
The experiment can be conducted with ```9_16_20```, ```9_16_40``` and ```9_16_80``` for ```fc_hw_dim``` respectively.
```
python main.py -e 300 --dataset xx --dataset_basrec xx --outf xx \
    --fea_hw_dim 9_16_20 --expansion 1 --lower_width 96 --strides 5 3 2 2 2 \
    --reduction 2 --norm in --act gelu \
    --view_num xx --frame_perview xx --basic_ind xx --resol 1920 1080 \
    --warmup 0.2 -b 1 --lr 0.0005 --qbit -1
```

### Evaluation experiments
To evaluate pre-trained model, just add ```--eval_only``` and specify model path with ```--weight```.
```
python main.py -e 300 --dataset xx --dataset_basrec xx --outf xx \
    --fea_hw_dim 9_16_20 --expansion 1 --lower_width 96 --strides 5 3 2 2 2 \
    --reduction 2 --norm in --act gelu \
    --view_num xx --frame_perview xx --basic_ind xx --resol 1920 1080 \
    --warmup 0.2 -b 1 --lr 0.0005 --qbit -1 \
    --weight xx --eval_only
```

### Decoding: Dump predictions with pre-trained model 
To dump predictions with pre-trained model, just add ```--dump_images``` besides ```--eval_only``` and ```--weight```
```
python main.py -e 300 --dataset xx --dataset_basrec xx --outf xx \
    --fea_hw_dim 9_16_20 --expansion 1 --lower_width 96 --strides 5 3 2 2 2 \
    --reduction 2 --norm in --act gelu \
    --view_num xx --frame_perview xx --basic_ind xx --resol 1920 1080 \
    --warmup 0.2 -b 1 --lr 0.0005 --qbit -1 \
    --weight xx --eval_only --dump_images
```

## Model Compression Experiments

Step1: Train the model without prune and use the parameters: ```--expansion 8 --lower_width 24```

```
python main.py -e 300 --dataset xx --dataset_basrec xx --outf xx \
    --fea_hw_dim 9_16_20 --expansion 8 --lower_width 24 --strides 5 3 2 2 2 \
    --reduction 2 --norm in --act gelu \
    --view_num xx --frame_perview xx --basic_ind xx --resol 1920 1080 \
    --warmup 0.2 -b 1 --lr 0.0005 --qbit 8 \
```


Step2: Train the model with prune, based on model of Step1
```
python main.py -e 300 --dataset xx --dataset_basrec xx --outf xx \
    --fea_hw_dim 9_16_20 --expansion 8 --lower_width 24 --strides 5 3 2 2 2 \
    --reduction 2 --norm in --act gelu \
    --view_num xx --frame_perview xx --basic_ind xx --resol 1920 1080 \
    --warmup 0.2 -b 1 --lr 0.0005 --qbit 8 \
    --weight Step1.pth --prune_ratio 0.4 --not_resume_epoch \
```

Step3: Evaluating on the pruned model
```
python main.py -e 300 --dataset xx --dataset_basrec xx --outf xx \
    --fea_hw_dim 9_16_20 --expansion 8 --lower_width 24 --strides 5 3 2 2 2 \
    --reduction 2 --norm in --act gelu \
    --view_num xx --frame_perview xx --basic_ind xx --resol 1920 1080 \
    --warmup 0.2 -b 1 --lr 0.0005 --qbit 8 \
    --weight Step2_pruned.pth --prune_ratio 0.4 \
    --eval_only \
```

## Acknowledgement

This repository is based on [NeRV](https://github.com/haochen-rye/NeRV).
