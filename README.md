# Geo-Localization

 
## Requirements
All requirements are listed in the `requirements.txt`. Please use the following command to install all required packages in an individual environment:

```bash
# clone this repo
git clone https://github.com/ewanowara/Geo-Localization.git && cd Geo-Localization

# Install dependencies
python3 -m venv glimpse_classification_env
pip install -r requirements.txt
source glimpse_classification_env/bin/activate 

```

## Reproduce Results

#### Test on Already Trained Model

To use the pre-trained model by default, first download the model checkpoint by running:

```
mkdir -p saved_models
wget    saved_models/
```

Inference with pre-trained model:

```bash
python Test_ViT.py --cuda_base cuda:0
```

Available argparse parameter:
```bash
--seg_channels
  number of semantic segmentation channels (default: 3)

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
```

#### Train from Scratch on Already Trained Model

```bash
python Train_ViT.py --cuda_base cuda:0
```

#### download and preprocess training and validation images
```bash
wget https://github.com/TIBHannover/GeoEstimation/releases/download/v1.0/mp16_urls.csv -O resources/mp16_urls.csv
wget https://github.com/TIBHannover/GeoEstimation/releases/download/pytorch/yfcc25600_urls.csv -O resources/yfcc25600_urls.csv 
python download_images.py --output resources/images/mp16 --url_csv resources/mp16_urls.csv --shuffle
python download_images.py --output resources/images/yfcc25600 --url_csv resources/yfcc25600_urls.csv --shuffle --size_suffix ""
```
