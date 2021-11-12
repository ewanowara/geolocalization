# Geo-Localization

 
## Requirements
All requirements are listed in the `requirements.txt`. Please use the following command to install all required packages in an individual environment:

### clone this repo
git clone  && cd 

### Install dependencies
```bash
 python3 -m venv glimpse_classification_env
 pip install -r requirements.txt
 source glimpse_classification_env/bin/activate 

```

## Reproduce Results

#### Test on Already Trained Model

To use the pre-trained model by default, first download the model checkpoint by running:

```
mkdir -p saved_models
wget  http...  saved_models/
```

Inference with pre-trained model:

```bash
python Test_ViT.py --cuda_base cuda:0
```

Available argparse parameter:
```
--seg_channels
  Number of Segmentation Channels (default: 3)

--scene_of_interest
  which scene categories to train the model on - indoor - 0, natural - 1, urban - 2, all - 3 (default: 3)
  
--cuda_base 
   which GPU ID to use to load the model (default: cuda:0)
  
 --device_ids
   which GPU IDs to use to train / validate the model (default: 0)
 
 --epochs 
   how many training epochs to use (default: 25)
   
 --s2_cells 
   which geo-cell partitioning to use for training and validation - coarse (largest cells - 3298 total), middle (7202 total), fine (smallest cells 12893 total)
   

#### Train from Scratch on Already Trained Model

```bash
python Train_ViT.py --cuda_base cuda:0
```
