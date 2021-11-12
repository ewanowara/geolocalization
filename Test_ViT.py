from dataloader import MultiPartitioningClassifier, cuda_base, device_ids
import yaml

with open('config/base_model.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

model_params = config["model_params"]
tmp_model = MultiPartitioningClassifier(hparams=Namespace(**model_params))

val_data_loader = tmp_model.val_dataloader()

import pandas as pd
import numpy as np
import torch
from PIL import Image
import torch.nn as nn
from torch.optim import lr_scheduler
from transformers import ViTModel, ViTConfig
import warnings
warnings.filterwarnings("ignore", message="numerical errors at iteration 0")

from torch.utils.data import DataLoader

import geopy.distance
import glob

num_classes_coarse = 3298
num_classes_middle = 7202
num_classes_fine = 12893
learning_rate = config["lr"]

num_scene = 3 

s2cell_path = 'resources/s2_cells/cells_50_1000.csv'
s2cells = pd.read_csv(s2cell_path) 

num_digits = 2 # how many digits to round up to for printing the results

use_segmentation = True
seg_channels  = 3
num_scenes = 3

class GeoClassification(nn.Module):

    def __init__(self):
        super(GeoClassification, self).__init__()
        
        self.configuration_RGB = ViTConfig(num_channels = 3)
        self.configuration_Seg = ViTConfig(num_channels = seg_channels)
        self.vit_RGB = ViTModel(self.configuration_RGB).from_pretrained('google/vit-base-patch16-224-in21k')
        self.vit_Seg = ViTModel(self.configuration_Seg).from_pretrained('google/vit-base-patch16-224-in21k')
        
        self.cmaf_1 = nn.Linear(self.vit_RGB.config.hidden_size, 8)
        self.cmaf_2 = nn.Linear(8, 1)

        self.dropout = nn.Dropout(p=0.20)
        self.relu = torch.nn.LeakyReLU()

        self.classifier_1 = nn.Linear(self.vit_RGB.config.hidden_size, num_classes_fine)
        self.classifier_2 = nn.Linear(self.vit_RGB.config.hidden_size, num_scenes)
    
    def forward(self, rgb_image, seg_image):

        seg_image = torch.permute(seg_image, (0,3,1,2))
        
        outputs_RGB = self.vit_RGB(rgb_image).last_hidden_state
        outputs_Seg = self.vit_Seg(seg_image).last_hidden_state
        outputs_RGB = outputs_RGB[:,0,:]
        outputs_Seg = outputs_Seg[:,0,:]

        weight_RGB = self.cmaf_2(self.dropout(self.relu(self.cmaf_1(outputs_RGB))))
        weight_Seg = self.cmaf_2(self.dropout(self.relu(self.cmaf_1(outputs_Seg))))

        outputs_RGB = torch.mul(outputs_RGB, weight_RGB)
        outputs_Seg = torch.mul(outputs_Seg, weight_Seg)

        outputs = torch.add(outputs_RGB, outputs_Seg)
        geo_logits = self.classifier_1(outputs)
        scene_logits = self.classifier_2(outputs)

        return geo_logits, scene_logits


device = torch.device(cuda_base if torch.cuda.is_available() else 'cpu')
model = GeoClassification()     
model = model.to(device)
model = nn.DataParallel(model, device_ids=device_ids)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum = config["momentum"], weight_decay = config["weight_decay"])
step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = config["milestones"], gamma= config["gamma"])

model_path = 'saved_models/Saved_RGB_ViT.tar' # will be updated once the final model has converged

saved_model = torch.load(model_path, map_location='cuda:0')
model.load_state_dict(saved_model['Model_state_dict'])
optimizer.load_state_dict(saved_model['optimizer_state_dict'])

model.to(device)

d_val = pd.read_csv('resources/yfcc25600_places365.csv')
LAT1 = s2cells['latitude_mean']  # ground truth
LON1 = s2cells['longitude_mean']  

LAT_val = d_val['LAT']  
LON_val = d_val['LON']  

km_distance = []
with torch.no_grad():
    
    n_correct = 0
    n_samples = 0

    for i, (rgb_image, seg_image, label, _, _, scene_key, img_name) in enumerate(val_data_loader):

        rgb_image = rgb_image.type(torch.float32).to(device)
        seg_image = seg_image.type(torch.float32).to(device)

        label = label[0].to(device)

        # Forward pass
        model.eval()
        outputs, _ = model(rgb_image, seg_image)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)

        n_samples += label.size(0)
        n_correct += (predicted == label).sum().item()

        target_total_test.append(label)
        predicted_total_test.append(predicted)
        model_outputs_total_test.append(outputs)

        target_inter = [t.cpu().numpy() for t in target_total_test]
        predicted_inter = [t.cpu().numpy() for t in predicted_total_test]
        outputs_inter = [t.cpu().numpy() for t in model_outputs_total_test]

        target_inter =  np.stack(target_inter, axis=0).ravel()
        predicted_inter =  np.stack(predicted_inter, axis=0).ravel()
        outputs_inter = np.concatenate(outputs_inter, axis=0)

        predicted = predicted.detach()
        predicted = predicted.cpu()
        predicted = predicted.numpy()

        scene_key = scene_key.detach()
        scene_key = scene_key.cpu()
        scene_key = scene_key.numpy()

        # for each image
        for ii in range(len(rgb_image)):     
            # covert geo cell class to longitude and latitude # TODO: 
            class_idx = np.array(s2cells['class_label'] == predicted[ii])

            pred_lat = LAT1[class_idx].values 
            pred_lon = LON1[class_idx].values 

            class_GT_idx = np.array(d_val['IMG_ID'] == img_name[ii])

            label_lat = LAT_val[class_GT_idx].values 
            label_lon = LON_val[class_GT_idx].values 

            km_distance_i = geopy.distance.great_circle((label_lat[0], label_lon[0]), (pred_lat[0], pred_lon[0])).km
            km_distance.append(km_distance_i)
            km_distance_all_scenes2.append(km_distance_i)    
            
km_distance = np.array(km_distance)
    
# print results in terms of the geographic Great Circle Distance betweeen the predicted and ground truth GPS positons    
print('1 km & 25 km & 200 km & 750 km & 2,500 km ' + "\\\\" ) # print in the Latex table format
print(str(round(100*(len(km_distance[km_distance<=1])) / len(km_distance),num_digits)) + ' & ' + str(round(100*(len(km_distance[km_distance<=25])) / len(km_distance),num_digits)) + ' & ' + str(round(100*(len(km_distance[km_distance<=250])) / len(km_distance),num_digits)) + ' & ' + str(round(100*(len(km_distance[km_distance<=750])) / len(km_distance),num_digits)) + ' & ' + str(round(100*(len(km_distance[km_distance<=2500])) / len(km_distance),num_digits))  + " \\\\"  )

# print results in terms of the classification accuracy (predicted geo-cell class)
print(f' Accuracy of the network on the test set with the saved model is: {accuracy_score(target_inter, predicted_inter)}')
print(f' Top 2 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=2)}')
print(f' Top 5 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=5)}')
print(f' Top 10 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=10)}')
print(f' Top 50 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=50)}')
print(f' Top 100 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=100)}')
print(f' Top 200 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=200)}')
print(f' Top 300 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=300)}')
print(f' Top 500 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=500)}')