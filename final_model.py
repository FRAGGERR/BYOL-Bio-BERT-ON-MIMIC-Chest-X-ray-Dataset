import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
import logging
import time
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
import yaml
# from data import DataLoader as CustomDataLoader
from data import DataLoader
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.cuda.empty_cache()

# Initialize logging
logging.basicConfig(filename='training.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Device configuration
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Load configuration
config_file = "config1.yaml"
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)
config['data_pct'] = 100

# Data loading
data_ins   = DataLoader(config)
# train_loader, valid_loader, test_loader = data_ins.GetMimicDataset() #you are not supposed to use the mimic dataset
#you need to use the multimodal data
train_loader, valid_loader = data_ins.GetMultimodalPretrainingDataset()

# Define custom BYOL model
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(ProjectionHead, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.block(x)

class PredictionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(PredictionHead, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.block(x)

class BYOL(nn.Module):
    def __init__(self, backbone):
        super(BYOL, self).__init__()
        self.backbone = backbone
        self.projection_head = ProjectionHead(2048, 4096, 256)
        self.prediction_head = PredictionHead(256, 4096, 256)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        for param in self.backbone_momentum.parameters():
            param.requires_grad = False
        for param in self.projection_head_momentum.parameters():
            param.requires_grad = False

    def forward_online(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

#     def forward_momentum(self, x):
    def forward_target(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z
    
    def forward(self, x):
        online = self.forward_online(x)
        target = self.forward_target(x)
        
        return online, target
    

def negative_cosine_similarity(p, z):
    return -F.cosine_similarity(p, z.detach(), dim=-1).mean()

def vicreg_loss(x, y, sim_weight=25.0, var_weight=25.0, cov_weight=1.0):
#     repr_loss = F.mse_loss(x, y)

    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)
    
    std_x = torch.sqrt(x.var(dim=0) + 1e-4)
    std_y = torch.sqrt(y.var(dim=0) + 1e-4)
    std_loss = (torch.mean(F.relu(1 - std_x)) + torch.mean(F.relu(1 - std_y))) * var_weight
    
    cov_x = (x.T @ x) / (x.size(0) - 1)
    cov_y = (y.T @ y) / (y.size(0) - 1)
    cov_loss = (off_diagonal(cov_x).pow_(2).sum() + off_diagonal(cov_y).pow_(2).sum()) * cov_weight
    
    return std_loss + cov_loss

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

# BYOL
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
backbone = nn.Sequential(*list(resnet.children())[:-1]).to(device) ##added .to(device)
byol_model = BYOL(backbone).to(device)

class TextProjectionHead(nn.Module):
    def __init__(self):
        super().__init__()
        embedding_dim= 768
        projection_dim=256
        dropout=0.2
        
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu       = nn.GELU()
        self.fc         = nn.Linear(projection_dim, projection_dim)
        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class TextEncoder(nn.Module):
    def __init__(self ):
        super().__init__()
        
        model_name="emilyalsentzer/Bio_ClinicalBERT"
        pretrained=True
        trainable=False
        
        if pretrained:
            self.model = AutoModel.from_pretrained(model_name)
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0
        self.projection = TextProjectionHead()
        for p in self.projection.parameters():
            p.requires_grad = True

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.model(input_ids=input_ids, attention_mask=attention_mask,return_dict=False)
        text_embeddings   = self.projection(pooled_output)
        return text_embeddings

class CombinedModel(nn.Module):
    def __init__(self, image_model, text_model):
        super(CombinedModel, self).__init__()
        self.image_model = image_model
        self.text_model = text_model
        # self.fc = nn.Linear(256, 1)##changed
    
    def forward(self, images, input_ids, attention_mask):
        online, target = self.image_model(images)
        text_features = self.text_model(input_ids,attention_mask)
        # outputs = self.fc(text_features) ##changed
        return online, target, text_features

biobert_model = TextEncoder()
combined_model = CombinedModel(byol_model, biobert_model).to(device)

# Training and validation
num_epochs = 300
learning_rate = 0.001
optimizer = torch.optim.Adam(combined_model.parameters(), lr=learning_rate)
classification_criterion = nn.BCELoss()

# Training loop for the combined model
total_start_time = time.time()
roc_auc_scores = []


# Training loop for the combined model
total_start_time = time.time()
roc_auc_scores = []

for epoch in range(num_epochs):
    combined_model.train()
    epoch_loss = 0
    for batch in tqdm(train_loader):
        input_ids = batch['caption_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        images = batch['imgs']
        view_1, view_2 = images
        view_1 = view_1.to(device)
        view_2 = view_2.to(device)

        optimizer.zero_grad()

        # Ensure you are correctly unpacking the outputs from the combined_model forward pass
        online_1, target_1, text_features_1 = combined_model(view_1, input_ids, attention_mask)
        online_2, target_2, text_features_2 = combined_model(view_2, input_ids, attention_mask)

        # Calculate BYOL losses
        loss_byol = (negative_cosine_similarity(online_1, target_2) + negative_cosine_similarity(online_2, target_1)) / 2

        # Calculate VICReg variance losses
        variance_I = vicreg_loss(online_1, online_2)
        variance_T = vicreg_loss(text_features_1, text_features_2)
        loss_vicreg = F.mse_loss(variance_I, variance_T)

        # Combined loss
        loss = (loss_byol + loss_vicreg) / 2
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}")

total_end_time = time.time()
total_training_time = total_end_time - total_start_time
logging.info(f"Total training time: {total_training_time:.2f seconds}")

# Save the model checkpoint
torch.save(combined_model.state_dict(), "combined_model.pth")
