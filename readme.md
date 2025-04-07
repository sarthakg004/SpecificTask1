# Text Layout Detection Model

This repository contains my solution for specific task I for Renaissance project for HumanAI GSOC 2025 selection. 

## Project Overview

The project follows a complete machine learning workflow:
1. Data preprocessing and generation
2. Mask creation for segmentation
3. Model architecture design
4. Training pipeline implementation 
5. Evaluation and visualization

## Data Generation

### Preprocessing Pipeline

The preprocessing pipeline includes the following steps:

1. **PDF to Images**: Converting PDF documents to images using PyMuPDF (fitz)

2. **Image Transformation**: Each page image undergoes several transformations:
   - Skew correction
   - Image normalization
   - Resolution adjustment to ensure 300 PPI

3. **Standardization**: Images are resized and padded to a standard format (512×384 pixels)

### Mask Generation

For each document image, a mask is generated using OCR to identify text regions:

```python
def generate_mask(image_path):
    image = cv2.imread(image_path)
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # OCR to find text regions
    data = pytesseract.image_to_data(thresh, config='--psm 6', output_type=pytesseract.Output.DICT)
    
    # Group words into sentences based on position
    sentences = []
    # ... grouping logic ...
    
    # Draw text regions on mask
    for sentence in sentences:
        # Find bounding box
        # ... bounding box calculation ...
        cv2.rectangle(mask, (min_x, min_y), (max_x, max_y), 255, -1)
    
    return mask
```

The process:
1. Uses Tesseract OCR to detect all text elements
2. Groups detected words into sentences based on line position
3. Creates bounding boxes around text regions
4. Draws filled rectangles on a binary mask for each text region

## Model Architecture

```mermaid
flowchart LR
    classDef encoder fill:#ffdddd,stroke:#333,stroke-width:1px,color:#333
    classDef decoder fill:#ddddff,stroke:#333,stroke-width:1px,color:#333
    classDef bottleneck fill:#ffffdd,stroke:#333,stroke-width:1px,color:#333
    classDef skip stroke:#ff9999,stroke-width:2px
    classDef deep stroke:#9999ff,stroke-width:2px,stroke-dasharray: 5 5
    
    %% Main flow
    input([Image]) --> encoder[Encoder Path]:::encoder
    encoder --> bottleneck[Bottleneck]:::bottleneck
    bottleneck --> decoder[Decoder Path]:::decoder
    decoder --> output([Segmentation Mask])
    
    %% Skip connections
    encoder -- Skip Connections<br/>with Attention Gates --> decoder
    
    %% Deep supervision
    decoder -. Deep Supervision .-> aux([Auxiliary Outputs])
    
    %% Loss
    output & aux --> loss([Combined Loss])
    
    %% Style the connections
    linkStyle 2 stroke:#ff9999,stroke-width:2px;
    linkStyle 3 stroke:#9999ff,stroke-width:2px,stroke-dasharray: 5 5;

```
The model uses a U-Net architecture with attention mechanisms to identify text areas in document pages.

### Enhanced U-Net with Attention

The model is based on a U-Net architecture with several enhancements:

1. **Attention Gates**: Focus on relevant features during decoding

2. **Deep Supervision**: Multiple output paths at different scales

3. **Residual Connections**: Improving gradient flow

4. **Dropout Regularization**: Preventing overfitting

Key components:
- **DoubleConv**: Double convolution blocks with LeakyReLU activation
- **Down**: Downsampling paths with optional residual connections
- **Up**: Upsampling paths with attention gates
- **Bottleneck**: Feature channel expansion with dropout

## Model Training

### Custom Loss Function

A combined loss function to handle the challenges of text segmentation:

```python
class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.3, focal_weight=0.2, ds_weight=0.3):
        # ... initialization ...
    
    def dice_loss(self, pred, target, smooth=1.0):
        # ... soft dice loss implementation ...
    
    def focal_loss(self, pred, target, alpha=0.25, gamma=2.0):
        # ... focal loss implementation ...
    
    def forward(self, outputs, target):
        # ... combines multiple losses with deep supervision handling ...
```

The loss combines:
1. **Dice Loss**: Better handling of class imbalance
2. **Binary Cross-Entropy**: Standard segmentation loss
3. **Focal Loss**: Emphasizes hard examples
4. **Deep Supervision Loss**: Additional gradient pathways

### Optimization

- **Adam optimizer** with weight decay for regularization
- **Learning rate scheduler** to reduce LR on plateaus
- **Early stopping** to prevent overfitting

<img src="model_output/model1/training_history_plot.png" alt="metrics" width="1000"/>


## Evaluation Metrics

The model is evaluated using standard segmentation metrics:

1. **IoU (Intersection over Union)**: Measures overlap between predicted and true masks
2. **Dice Coefficient (F1 Score)**: Harmonic mean of precision and recall
3. **Precision**: Ratio of true positives to all positive predictions
4. **Recall**: Ratio of true positives to all actual positives
5. **Accuracy**: Overall pixel-wise accuracy

| Metric         | Train        | Validation    | Test       |
|----------------|--------------|----------------|------------|
| Loss           | 0.2437       | 0.2428         | 0.2241     |
| IoU            | 0.7761       | 0.7841         | 0.7859     |
| Dice/F1        | 0.8726       | 0.8785         | 0.8791     |
| Accuracy       | —            | —              | 0.8806     |
| Precision      | —            | —              | 0.8879     |
| Recall         | —            | —              | 0.8716     |

<img src="model_output/model1/prediction_visualization.png" alt="metrics" width="1000"/>

## Requirements

- Python 3.7+
- PyTorch
- OpenCV
- PyMuPDF (fitz)
- pytesseract
- Tesseract OCR engine
- segmentation_models_pytorch
- pandas
- matplotlib
- scikit-learn
- tqdm

## Directory Structure

```
data/
├── 1_raw/            # Raw PDF files
├── 2_splitted/       # Images extracted from PDFs
├── 3_transformed/    # Preprocessed images
├── 4_final/          # Standardized images
└── masks/            # Generated segmentation masks

model_output/
└── model1/           # Trained model weights and logs
```
