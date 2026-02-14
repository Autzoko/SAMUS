# SAMUS
This repo is the official implementation for:\
[SAMUS: Adapting Segment Anything Model for Clinically-Friendly and Generalizable Ultrasound Image Segmentation.](https://arxiv.org/pdf/2309.06824.pdf)\
(The details of our SAMUS can be found in the models directory in this repo or in the paper.)

## Highlights
🏆 Low GPU requirements. (one 3090ti with 24G GPU memory is enough)\
🏆 Large ultrasound dataset. (about 30K images and 69K masks covering 6 categories)\
🏆 Excellent performance, especially in generalization ability.\
✨ We have released the pre-trained model in [SAMUS](https://drive.google.com/file/d/1nQjMAvbPeolNpCxQyU_HTiOiB5704pkH/view?usp=sharing).\
✨ We have released the preprocessed dataset in [US30K](https://drive.google.com/file/d/13MUXQIyCXqNscIKTLRIEHKtpak6MJby_/view?usp=sharing).

## Installation
Following [Segment Anything](https://github.com/facebookresearch/segment-anything), `python=3.8.16`, `pytorch=1.8.0`, and `torchvision=0.9.0` are used in SAMUS.

1. Clone the repository.
    ```
    git clone https://github.com/xianlin7/SAMUS.git
    cd SAMUS
    ```
2. Create a virtual environment for SAMUS and activate the environment.
    ```
    conda create -n SAMUS python=3.8
    conda activate SAMUS
    ```
3. Install Pytorch and TorchVision.
   (you can follow the instructions [here](https://pytorch.org/get-started/locally/))
5. Install other dependencies.
  ```
    pip install -r requirements.txt
  ```
## Checkpoints
- We use checkpoint of SAM in [`vit_b`](https://github.com/facebookresearch/segment-anything) version.
- The trained SAMUS can be downloaded [here](https://drive.google.com/file/d/1nQjMAvbPeolNpCxQyU_HTiOiB5704pkH/view?usp=sharing).

## Data
- US30K consists of seven publicly-available datasets, including [TN3K]( https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation), [DDTI]( https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation), [TG3K](https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation), [BUSI](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset), [UDIAT](http://www2.docm.mmu.ac.uk/STAFF/M.Yap/dataset.php), [CAMUS](http://camus.creatis.insa-lyon.fr/challenge/), and [HMC-QU](https://aistudio.baidu.com/aistudio/datasetdetail/102406).
- The preprocessed US30K can be downloaded [here](https://drive.google.com/file/d/13MUXQIyCXqNscIKTLRIEHKtpak6MJby_/view?usp=sharing).
- All images were saved in PNG format. No special pre-processed methods are used in data preparation.
- We have provided some examples to help you organize your data. Please refer to the file folder [example_of_required_dataset_format](https://github.com/xianlin7/SAMUS/tree/main/example_of_required_dataset_format).\
  Specifically, each line in train/val.txt should be formatted as follows:
  ```
    <class ID>/<dataset file folder name>/<image file name>
  ```
- The relevant information of your data should be set in [./utils/config.py](https://github.com/xianlin7/SAMUS/blob/main/utils/config.py) 

## Training
Once you have the data ready, you can start training the model.
```
cd "/home/...  .../SAMUS/"
python train.py --modelname SAMUS --task <your dataset config name>
```
## Testing
Do not forget to set the load_path in [./utils/config.py](https://github.com/xianlin7/SAMUS/blob/main/utils/config.py) before testing.
```
python test.py --modelname SAMUS --task <your dataset config name>
```

## AutoSAMUS on Breast Ultrasound Datasets

Train AutoSAMUS (prompt-free variant) on 5 breast ultrasound datasets: BUSI, BUSBRA, BUS, BUS\_UC, BUS\_UCLM.

### 1. Preprocess Data

Convert breast datasets from UltraSAM COCO format to SAMUS format (256x256 grayscale):

```bash
python scripts/preprocess_breast_for_samus.py \
    --ultrasam-data ../UltraSam/UltraSAM_DATA \
    --save-dir SAMUS_DATA
```

This creates `SAMUS_DATA/` with split files in `MainPatient/` (2705 train, 678 val).

### 2. Download Pretrained SAMUS Checkpoint

```bash
mkdir -p checkpoints
pip install gdown
gdown 1nQjMAvbPeolNpCxQyU_HTiOiB5704pkH -O checkpoints/samus_pretrained.pth
```

### 3. Train

**Train on all 5 breast datasets combined:**
```bash
python train.py \
    --modelname AutoSAMUS \
    --task AllBreast \
    --batch_size 8 \
    --base_lr 0.0005 \
    --warmup True --warmup_period 250 \
    -keep_log True
```

**Train on a single dataset:**
```bash
# Available tasks: BreastBUSI, BreastBUSBRA, BreastBUS, BreastBUS_UC, BreastBUS_UCLM
python train.py \
    --modelname AutoSAMUS \
    --task BreastBUSI \
    --batch_size 8 \
    --base_lr 0.0001 \
    --warmup True --warmup_period 250 \
    -keep_log True
```

**SLURM (NYU Greene HPC):**
```bash
sbatch scripts/sbatch_train_autosamus_allbreast.sh
```

Checkpoints are saved to `checkpoints/AllBreast/` (or `checkpoints/Breast<DATASET>/` for per-dataset).

### 4. Evaluate

```bash
python scripts/eval_autosamus_breast.py \
    --checkpoint checkpoints/AllBreast/AutoSAMUS_best.pth \
    --data-root SAMUS_DATA \
    --eval-datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM \
    --save-json result/AllBreast/eval_results.json
```

Reports per-dataset and combined metrics: Dice, IoU, HD95, ASD, Precision, Recall, Specificity, Accuracy.

**SLURM:**
```bash
sbatch scripts/sbatch_eval_autosamus.sh checkpoints/AllBreast/AutoSAMUS_best.pth
```

## Citation
If our SAMUS is helpful to you, please consider citing:
```
@misc{lin2023samus,
      title={SAMUS: Adapting Segment Anything Model for Clinically-Friendly and Generalizable Ultrasound Image Segmentation}, 
      author={Xian Lin and Yangyang Xiang and Li Zhang and Xin Yang and Zengqiang Yan and Li Yu},
      year={2023},
      eprint={2309.06824},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
