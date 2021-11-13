from dataloader_glimpse  import MultiPartitioningClassifier, seg_channels, cuda_base, server, scenes, device_ids, num_epochs
import yaml 
from argparse import Namespace
import torch
import argparse

with open('config/base_model.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

model_params = config["model_params"]
tmp_model = MultiPartitioningClassifier(hparams=Namespace(**model_params))

train_data_loader = tmp_model.train_dataloader()
val_data_loader = tmp_model.val_dataloader()
# Choose the first n_steps batches with 64 samples in each batch
n_steps = 2

import os
import pandas as pd
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,models
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score
from torchsummary import summary
import random
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from torchsummary import summary
from transformers import ViTModel, ViTConfig
import warnings
warnings.filterwarnings("ignore", message="numerical errors at iteration 0")

def topk_accuracy(target, output, k):
    topn = np.argsort(output, axis = 1)[:,-k:]
    return np.mean(np.array([1 if target[k] in topn[k] else 0 for k in range(len(topn))]))

num_classes_coarse = 3298
num_classes_middle = 7202
num_classes_fine = 12893
learning_rate = config["lr"]
num_scenes = int(scenes)

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

#print(summary(model, (3, 224, 224)))


import warnings
warnings.filterwarnings("ignore")

n_total_steps = len(train_data_loader)

batch_wise_loss = []
batch_wise_micro_f1 = []
batch_wise_macro_f1 = []
epoch_wise_top_10_accuracy = []
epoch_wise_top_50_accuracy = []
epoch_wise_top_100_accuracy = []
epoch_wise_top_200_accuracy = []
epoch_wise_top_300_accuracy = []
epoch_wise_top_500_accuracy = []

for epoch in range(num_epochs):
    for i, (rgb_image, seg_image, label, _, _, _) in enumerate(train_data_loader):
        
        rgb_image = rgb_image.type(torch.float32).to(device)
        seg_image = seg_image.type(torch.float32).to(device)
        label = label[2].to(device)
        
        # Forward pass
        model.train()
        outputs, _ = model(rgb_image, seg_image)
        loss = criterion(outputs, label)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step_lr_scheduler.step()
        
        # train only for a subset of the training set to test the code first
        if (i+1) % 2000 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

    target_total_test = []
    predicted_total_test = []
    model_outputs_total_test = []

    with torch.no_grad():
        
        n_correct = 0
        n_samples = 0

        for i, (rgb_image, seg_image, label, _, _, _) in enumerate(val_data_loader):
            
            rgb_image = rgb_image.type(torch.float32)
            seg_image = seg_image.type(torch.float32)
            label = label[2].to(device)

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

        current_top_10_accuracy = topk_accuracy(target_inter, outputs_inter, k=10)
        epoch_wise_top_10_accuracy.append(current_top_10_accuracy)
        current_top_50_accuracy = topk_accuracy(target_inter, outputs_inter, k=50)
        epoch_wise_top_50_accuracy.append(current_top_50_accuracy)
        current_top_100_accuracy = topk_accuracy(target_inter, outputs_inter, k=100)
        epoch_wise_top_100_accuracy.append(current_top_100_accuracy)
        current_top_200_accuracy = topk_accuracy(target_inter, outputs_inter, k=200)
        epoch_wise_top_200_accuracy.append(current_top_200_accuracy)
        current_top_300_accuracy = topk_accuracy(target_inter, outputs_inter, k=300)
        epoch_wise_top_300_accuracy.append(current_top_300_accuracy)
        current_top_500_accuracy = topk_accuracy(target_inter, outputs_inter, k=500)
        epoch_wise_top_500_accuracy.append(current_top_500_accuracy)
       
        print(f' Accuracy of the network on the test set after Epoch {epoch+1} is: {accuracy_score(target_inter, predicted_inter)}')
        print(f' Top 2 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=2)}')
        print(f' Top 5 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=5)}')
        print(f' Top 10 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=10)}')
        print(f' Top 50 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=50)}')
        print(f' Top 100 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=100)}')
        print(f' Top 200 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=200)}')
        print(f' Top 300 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=300)}')
        print(f' Top 500 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=500)}')
        
        print(f' Best Top_10_accuracy on test set till this epoch: {max(epoch_wise_top_10_accuracy)} Found in Epoch No: {epoch_wise_top_10_accuracy.index(max(epoch_wise_top_10_accuracy))+1}')
        print(f' Best Top_50_accuracy on test set till this epoch: {max(epoch_wise_top_50_accuracy)} Found in Epoch No: {epoch_wise_top_50_accuracy.index(max(epoch_wise_top_50_accuracy))+1}')
        print(f' Best Top_100_accuracy on test set till this epoch: {max(epoch_wise_top_100_accuracy)} Found in Epoch No: {epoch_wise_top_100_accuracy.index(max(epoch_wise_top_100_accuracy))+1}')
        print(f' Best Top_200_accuracy on test set till this epoch: {max(epoch_wise_top_200_accuracy)} Found in Epoch No: {epoch_wise_top_200_accuracy.index(max(epoch_wise_top_200_accuracy))+1}')
        print(f' Best Top_300_accuracy on test set till this epoch: {max(epoch_wise_top_300_accuracy)} Found in Epoch No: {epoch_wise_top_300_accuracy.index(max(epoch_wise_top_300_accuracy))+1}')
        print(f' Best Top_500_accuracy on test set till this epoch: {max(epoch_wise_top_500_accuracy)} Found in Epoch No: {epoch_wise_top_500_accuracy.index(max(epoch_wise_top_500_accuracy))+1}')
        print(f' Top_300_accuracy: {epoch_wise_top_300_accuracy}')
        print(f' Top_500_accuracy: {epoch_wise_top_500_accuracy}')

        if(current_top_500_accuracy == max(epoch_wise_top_500_accuracy)):
            torch.save({'Model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, './saved_models/Natural_RGB+Seg_Late_Fusion_Multitask_3_scene_ViT_fine.tar')


print("======================================")
print("Training Completed, Evaluating the test set using the best model")
print("======================================")


#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
saved_model = torch.load('./saved_models/Saved_RGB_ViT.tar')
model.load_state_dict(saved_model['Model_state_dict'])
optimizer.load_state_dict(saved_model['optimizer_state_dict'])

model.to(device)

target_total_test = []
predicted_total_test = []
model_outputs_total_test = []


with torch.no_grad():

        n_correct = 0
        n_samples = 0

        for i, (rgb_image, seg_image, label, _, _, _) in enumerate(val_data_loader):

            rgb_image = rgb_image.type(torch.float32).to(device)
            seg_image = seg_image.type(torch.float32).to(device)
            label = label[2].to(device)

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

        print(f' Accuracy of the network on the test set with the saved model is: {accuracy_score(target_inter, predicted_inter)}')
        print(f' Top 2 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=2)}')
        print(f' Top 5 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=5)}')
        print(f' Top 10 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=10)}')
        print(f' Top 50 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=50)}')
        print(f' Top 100 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=100)}')
        print(f' Top 200 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=200)}')
        print(f' Top 300 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=300)}')
        print(f' Top 500 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=500)}')
