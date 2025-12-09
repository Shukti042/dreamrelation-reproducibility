## Install the packages
Install the requirements metioned in the `requirements.txt`. I faced difficulty due to package incompatibility issues while running `pip install -r requirements.txt`, hence I installed them separately.

## Model Preparation
```bash
export HF_ENDPOINT=https://hf-mirror.com
mkdir checkpoints
huggingface-cli download doge1516/MS-Diffusion --local-dir ./checkpoints/MS-Diffusion
```

## Download LoRA Weights

### Official DreamRelation LoRA Weights
Download the LoRA weights released by the authors from [here](https://huggingface.co/QingyuShi/DreamRelation).

### Trained LoRA Weights (This Project)
Download the LoRA weights trained as part of this project from [here](https://pennstateoffice365-my.sharepoint.com/:u:/g/personal/nks5814_psu_edu/IQADpVr-cvlgTJD18M3Gh6W-ATtcRUIh0Il7Hj6MTg3iWdA?e=kuCaqB).

---

## Download Dataset

Download the **RelationBench** dataset from [here](https://huggingface.co/datasets/QingyuShi/RelationBench).

**Note:** All downloaded models, LoRA weights, and datasets should be placed in the **main project directory**.

## Training
Change the `INSTANCE_DIR` and `OUTPUT_DIR` accordingly.
```bash
export PYTHONPATH=.
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export INSTANCE_DIR="./data/hug"
export OUTPUT_DIR="./lora-weights/lora-trained-xl-hugging-0.1kp-0.001cal-0.6"
export HF_ENDPOINT="https://hf-mirror.com"
python train.py  --pretrained_model_name_or_path=$MODEL_NAME --instance_data_dir=$INSTANCE_DIR  --output_dir=$OUTPUT_DIR  --mixed_precision="fp16" --resolution=1024 --train_batch_size=1 --gradient_accumulation_steps=4 --learning_rate=1e-4 --report_to="wandb" --lr_scheduler="constant" --lr_warmup_steps=0 --max_train_steps=500 --validation_epochs=1000  --seed="0"
```
## Evaluation
To evaluate the metrics on all the relations, run the script `benchmark_all_relation.py`. To evaluate the metrics on the training set, run the script `benchmark_training.py`. To test the results on individual relations, run `benchmark_individual_relation`. Modify the configs for data and model accordingly.
## Citation
```
@inproceedings{DreamRelation,
  title={DreamRelation: Bridging Customization and Relation Generaion},
  author={Qingyu Shi, Lu Qi, Jianzong Wu, Jinbin Bai, Jingbo Wang, Yunhai Tong, Xiangtai Li},
  booktitle={CVPR},
  year={2025}
}
```
