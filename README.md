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
--cuda_base 
   which GPU ID to use (default: cuda:0)

#### Train from Scratch on Already Trained Model

```bash
python Train_ViT.py --cuda_base cuda:0
```
