# hg_keras
## Train
#### MPII Data Preparation
- Download MPII Dataset and put its images under `data/mpii/images`
- The json `mpii_annotations.json` contains all of images' annotations including train and validation.

#### Train
```
python train.py --gpuID 0 --epochs 100 --batch_size 24 --num_stack 2 --model_path ../../trained_models/hg
```
