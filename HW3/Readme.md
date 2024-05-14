# CUDA
cuda 12.1 (pytorch not yet support 12.4)
https://developer.nvidia.com/cuda-12-1-0-download-archive
# Conda 
```
###create a virtual environment with python version 3.10 
conda create -n <env_name> python=3.10
###Active
conda activate <env_name>
###cd into main folder and
python -m pip install e .
#install requirements
cd /example/text_to_image
python -m pip install -r requirements.txt
```

and might need to install xformers

```
python -m pip install xformers
```

### Create  dataset
create a directory to store data
```powershell
mkdir <directoryName>
```

put images which you want to generate with
and write a metadata.jsonl in directory folder
```json
{"file_name": "XXX.png", "text": "describe image"}
```

### Run 
```python
python train_text_to_image_lora.py --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" --dataset_name="..\..\Data\data" --dataloader_num_workers=0 --resolution=512 --center_crop --random_flip --train_batch_size=1 --gradient_accumulation_steps=4 --max_train_steps=240 --learning_rate=1e-04 --max_grad_norm=1 --lr_scheduler="cosine" --lr_warmup_steps=0 --output_dir="..\..\output" --report_to=wandb --use_8bit_adam --adam_beta1=0.9 --adam_weight_decay=1e-2 --validation_prompt="Maxwell the Cat" --seed=1337 --allow_tf32 --mixed_precision=fp16
```
#### if device is CPU instead of cuda, try:

```powershell
python -m pip uninstall torch
python -m pip install pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

