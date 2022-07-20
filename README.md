# Learnable-Sampling
This is the codebase for the ACM Multi-Media 2022 paper "3D Human Mesh Reconstruction by Learning to Sample Joint Adaptive Tokens for Transformers".
## Installation
1. set up the environment:
- python: 3.7.12
- pytorch: 1.7.1
- torchvision: 0.8.2
- cuda: 11.0
- pyrender: 0.1.45
- pyopengl: 3.1.6
- joblib: 1.1.0
- pytorch-lightning: 1.1.8
- opencv-python: 4.5.4.60
- pillow: 9.1.1
- loguru: 0.5.3
- yacs: 0.1.8
- scikit-image: 0.19.0
- azureml: 0.2.7
- azureml-core: 1.36.0.post2
- bottleneck

2. run the following command in the ./Learnable-Sampling directory to install:
```shell
python setup.py build develop
```

3. install opendr via pip+git:
```shell
pip install git+https://gitlab.eecs.umich.edu/ngv-python-modules/opendr.git
```

## Download SMPL data
Please prepare the SMPL data following [MeshGraphormer](https://github.com/microsoft/MeshGraphormer) and put all the data into ./src/modeling/data/. After doing so, the structure of the directory should be as follows:
```
$ src
|-- modeling
|   |-- data
|   |   |-- basicModel_f_lbs_10_207_0_v1.0.0.pkl
|   |   |-- basicModel_m_lbs_10_207_0_v1.0.0.pkl
|   |   |-- basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
|   |   |-- J_regressor_extra.npy
|   |   |-- J_regressor_h36m_correct.npy
```

## Download data and checkpoints
1. Download the tsv datasets following [METRO](https://github.com/microsoft/MeshTransformer/blob/main/docs/DOWNLOAD.md) and put the downloaded files into ./data/datasets/.
2. Download the db file and the evaluation config file for the 3DPW dataset from https://cloud.tsinghua.edu.cn/d/2da7d016fedb42de8039/ and put them into ./data/datasets/db/. Note that you should download the original images of the 3DPW dataset from the official webset and prepare them following the key 'image_name' in the db file.
3. Download the pretrained models from https://cloud.tsinghua.edu.cn/d/bc66b120bd5847649622/ and put them into ./data/pretrained_models/.
4. Download the trained checkpoints from https://cloud.tsinghua.edu.cn/d/61ba08aec1664798b7b6/ and put them into ./data/checkpoints/.
5. After doing the above four steps, the structure of the ./data directory should be as follows:
```
$ data  
|-- datasets  
|   |-- Tax-H36m-coco40k-Muco-UP-Mpii
|   |-- human3.6m
|   |-- coco_smpl
|   |-- muco
|   |-- up3d
|   |-- mpii
|   |-- 3dpw
|   |-- db
|   |   |-- 3dpw_test_db.pt
|   |   |-- test_3dpw.yaml
|-- pretrained_models
|   |-- pose_coco
|   |   |-- pose_hrnet_w32_256x192.pth
|   |   |-- pose_hrnet_w64_256x192.pth
|   |   |-- pose_resnet_50_256x192.pth
|   |-- pare_checkpoint.ckpt
|-- checkpoints
|   |-- 3dpw_checkpoint.bin
|   |-- h36m_checkpoint.bin
```
## Evaluation
To evaluate on 3DPW dataset, please run the following command:
```
python src/tools/main.py \
       --num_workers 1 \
       --config_yaml experiments/3dpw_config.yaml \
       --val_yaml data/datasets/db/test_3dpw.yaml \
       --per_gpu_eval_batch_size 10 \
       --output_dir ./output/ \
       --run_eval_only \
       --resume_checkpoint data/checkpoints/3dpw_checkpoint.bin
```
To evaluate on H36M dataset, please run the following command:
```
python src/tools/main.py \
       --num_workers 1 \
       --config_yaml experiments/h36m_config.yaml \
       --val_yaml data/datasets/human3.6m/valid.protocol2.yaml \
       --per_gpu_eval_batch_size 10 \
       --output_dir ./output/ \
       --run_eval_only \
       --resume_checkpoint data/checkpoints/h36m_checkpoint.bin
```
## Training
Training instructions will come soon.
## Acknowledgement
Our codebase is built upon open-source GitHub repositories. We thank all the authors for making their codes avaliable to facilitate the progress of our project.

[microsoft / MeshGraphormer](https://github.com/microsoft/MeshGraphormer)

[microsoft / MeshTransformer](https://github.com/microsoft/MeshTransformer)

[mkocabas / PARE](https://github.com/mkocabas/PARE)

[mkocabas / VIBE](https://github.com/mkocabas/VIBE)
