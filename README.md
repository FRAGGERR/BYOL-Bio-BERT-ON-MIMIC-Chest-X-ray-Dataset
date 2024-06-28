---

# Combined Model with ResNet-50 and BioBERT

This repository contains the code to train a combined model using ResNet-50 and BioBERT for multi-modal data. The model leverages BYOL (Bootstrap Your Own Latent) for self-supervised learning on images and integrates it with BioBERT for textual data processing.

## Requirements

- Python 3.8 or higher
- PyTorch 1.8 or higher
- torchvision 0.9 or higher
- scikit-learn
- transformers
- matplotlib
- pyyaml
- tqdm

You can install the necessary packages using the following command:
```sh
pip install -r requirements.txt
```

## Dataset

Ensure you have the MIMIC dataset or a similar dataset that includes both image and text data. The `data.py` file contains the custom data loader to handle the dataset.

## Configuration

Configuration settings are stored in a YAML file (`config1.yaml`). Update this file with your dataset paths and other parameters.

## Model Training

To train the combined model, follow these steps:

1. **Initialize Logging**
   Logging is configured to output to `training.log`. Adjust the logging level if needed:
   ```python
   logging.basicConfig(filename='training.log', level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
   ```

2. **Device Configuration**
   The code automatically detects and utilizes the GPU if available:
   ```python
   device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
   ```

3. **Data Loading**
   Load the dataset using the custom data loader:
   ```python
   data_ins = CustomDataLoader(config)
   train_loader, valid_loader, test_loader = data_ins.GetMimicDataset()
   ```

4. **Model Definition**
   Define the BYOL model based on ResNet-50:
   ```python
   base_encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
   byol_model = BYOL(base_encoder, hidden_dim=4096, projection_dim=256, num_classes=num_classes).to(device)
   ```

   Define the combined model with BioBERT:
   ```python
   combined_model = CombinedModel(byol_model, biobert_model, image_feature_dim, text_feature_dim).to(device)
   ```

5. **Training Loop**
   Train the model for the specified number of epochs:
   ```python
   for epoch in range(num_epochs):
       combined_model.train()
       epoch_loss = 0
       for batch in tqdm(train_loader):
           # Handle batch processing
           ...
           optimizer.zero_grad()
           outputs = combined_model(images, input_ids, attention_mask)
           loss = classification_criterion(outputs, labels)
           loss.backward()
           optimizer.step()
           epoch_loss += loss.item()
       ...
   ```

6. **Validation**
   Evaluate the model on the validation set after each epoch:
   ```python
   combined_model.eval()
   with torch.no_grad():
       for batch in tqdm(valid_loader):
           ...
           outputs = combined_model(images, input_ids, attention_mask)
           ...
   ```

7. **Save the Model**
   Save the trained model to a file:
   ```python
   torch.save(combined_model.state_dict(), "combined_model.pth")
   ```

8. **Plot Results**
   Plot the ROC AUC scores to visualize model performance:
   ```python
   plt.figure(figsize=(10, 8))
   for i in range(num_classes):
       plt.plot([roc_auc[i] for roc_auc in roc_auc_scores], label=f'Class {i}')
   plt.xlabel('Epoch')
   plt.ylabel('ROC AUC')
   plt.title('ROC AUC Scores per Epoch')
   plt.legend()
   plt.grid(True)
   plt.show()
   ```

## Acknowledgements

This code leverages the BYOL framework for self-supervised learning on images and integrates BioBERT for text processing. Thanks to the authors of these models and frameworks for their contributions to the field.

---
