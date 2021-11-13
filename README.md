# Geo-Localization

 
## Requirements
The code is written in PyTorch.

All requirements are listed in the `requirements.txt`. Please use the following command to install all required packages in an individual environment:

```bash
# clone this repo
git clone https://github.com/ewanowara/Geo-Localization.git && cd Geo-Localization

# Install dependencies
python3 -m venv glimpse_classification_env
pip install -r setup/requirements.txt
source glimpse_classification_env/bin/activate 

```

## Reproduce Results

#### Download and preprocess training and validation images
```bash
wget https://github.com/TIBHannover/GeoEstimation/releases/download/v1.0/mp16_urls.csv -O resources/mp16_urls.csv
wget https://github.com/TIBHannover/GeoEstimation/releases/download/pytorch/yfcc25600_urls.csv -O resources/yfcc25600_urls.csv 
python setup/download_images.py --output resources/images/mp16 --url_csv resources/mp16_urls.csv --shuffle
python setup/download_images.py --output resources/images/yfcc25600 --url_csv resources/yfcc25600_urls.csv --shuffle --size_suffix ""

# assign cell(s) for each image using the original meta information
wget https://github.com/TIBHannover/GeoEstimation/releases/download/v1.0/mp16_places365.csv -O resources/mp16_places365.csv
wget https://github.com/TIBHannover/GeoEstimation/releases/download/pytorch/yfcc25600_places365.csv -O resources/yfcc25600_places365.csv
python partitioning/assign_classes.py
# remove images that were not downloaded 
python setup/filter_by_downloaded_images.py
```

#### Test on Already Trained Model

To use the pre-trained model by default, first download the model checkpoint by running:

```
mkdir -p saved_models
wget https://www.dropbox.com/s/kqi5it6bexhzgmn/Saved_RGB_ViT.tar?dl=0   saved_models/Saved_RGB_ViT.tar
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

## Details About the Approach
The goal is to predict the geographic location from a single RGB image. We treat this geo-localization problem as a classification task, where we divide the Earh into a grid of geo-cell based on the GPS coordinates of the images, such that each geo-cell contains a similar number of images. 

We use a Vision Transformer (ViT) model as the backbone of our architecture. Very different features are important for different kinds of scenes, such as indoor scenes and outdoor urban or natural scenes. Hence, geo-localization can benefit from the knowledge about the scene, thus reducing the data space diversity and simplifying the classification problem. To achieve this, we train the ViT model with a multi-task learning objective and the model predicts the three coarse scene categories (natural, urban, indoor) for each image in addition to the geo-cell class. In order to improve the ViT model's ability to generalize to scenes captured in different conditions and with drastic variations in appearance, we jointly train the ViT model on the RGB images and their corresponding semantic segmentation maps. Semantic segmentation representation is largely invariant to significant appearance changes in the image, for example, caused by different lighting or weather, making the model more robust to these variations.

## Details About the Datasets

#### Training Set
We used the subset of the MediaEval Placing Task 2016 (MP-16) dataset to train our model. This dataset contains 4,654,532 geo-tagged images sourced from Flickr. These images were sourced without any restrictions which means that they contain very ambiguous scenes, such as photographs of food portraits of people.

#### Validation Set
We validate our model on the subset of the Yahoo Flickr Creative Commons 100 Million dataset (YFCC100M) which contains 25,600 images. The YFCC25600 is a subset of the MP-16 dataset without overlap in images or authors of the images. Similarly to MP-16, the YFCC25600 dataset contains images which were not originally intended for geo-localization and are as challenging as those in the MP-16 training dataset. 


