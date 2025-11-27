![ViT](https://github.com/user-attachments/assets/73764837-a38b-46ce-a043-95bee2e81a65)# Targeted False Positive Synthesis via Detector-guided Adversarial Diffusion Attacker for Robust Polyp Detection



╔══════════════════════════════════════════════════════════════════════╗
║                     Enhanced DADA Framework (2025)                   ║
║      Detector-Guided Adversarial Diffusion for Polyp Detection       ║
║             High-Value False-Positive Synthesis Pipeline             ║
╚══════════════════════════════════════════════════════════════════════╝



## Project Metadata
### Authors
- **Team:** Dalal Aldowaihi
- **Supervisor Name:** Dr. Muzammil Behzad
- **Affiliations:** IAU and KFUPM 

## Introduction
Traditional polyp detectors frequently produce false positives on complex colon backgrounds—such as fold textures, specular highlights, shadows, and circular lumen edges. These mistakes cause: Misleading clinical alerts, Increased cognitive load for gastroenterologists and Poor trust in AI tools
Existing generative methods mostly synthesize positive examples (polyp images).
However, medical detectors need challenging negative examples that look like polyps but are harmless background structures.

The Enhanced DADA Framework is the first diffusion-based adversarial pipeline designed to synthesize realistic false-positive samples to improve real-world robustness of colonoscopy detectors.


## Problem Statement
Colonoscopy detectors often mistake normal background artifacts for polyps due to: Specular highlights, Circular lumen edges, Wrinkles/folds, Unusual lighting and Texture similarity

These false positives degrade clinical performance.
The problem:
How can we synthesize realistic background artifacts that intentionally resemble polyps to reduce false positives?

## Application Area and Project Domain
**Application Area:** Medical imaging, Polyp detection, Robust deep learning, Adversarial augmentation

**Project Domain:** Computer Vision, Neural Networks, Deep Learning, Deep Learning Security, Adversarial Machine Learning. Subdomain of Robust Medical Image Classification under Adversarial Perturbations, Model Enhancement, Stable Diffusion, Adversarial diffusion framework, Data synthesis, Colorectal polyp detection


## What is the paper trying to do, and what are you planning to do?
DADA is a Detector-Guided Adversarial Diffusion Attacker, a novel framework designed to generate highly realistic false-positive examples for improving polyp detection models. The method works in three main steps:

**1. Background-Only Diffusion Model (BG-De)**
A modified DDPM is trained to learn only background patterns from colonoscopy images. Polyp regions are masked out during training, ensuring the model does not accidentally learn true polyp shapes.

**2. Adversarial Guidance (DADA)**
During image generation, adversarial gradients from a pre-trained detector (YOLO/DETR) are injected into the diffusion process.
These gradients guide the synthetic image toward patterns that fool the detector, intentionally creating challenging false positives.

**3. Local Inpainting for Realism**
Only a specific region of the image is modified, while the rest remains unchanged. This produces high-value, anatomically consistent false-positive samples.
The generated images look realistic and polyp-like but are actually background artifacts. Training detectors with these samples significantly reduces the number of false positives.

**Results**
Across public and private datasets, the method improves F1-score by +2.6% to +2.7%, outperforming adversarial attacks and other inpainting baselines.

This repository includes several research-level enhancements:
**Region-Adaptive loss function**
**Region-Adaptive Perturbation α** Instead of using the same adversarial factor (α) everywhere, the new version: Uses higher α in high-risk FP regions (folds, circular lumen), Uses lower α in smooth regions, Prevents over-perturbation and Increases biological plausibility
**Enhanced BG-De Training Stability** by using OneCycleLR learning rate scheduler.
  
### Project Documents
- **Presentation:** [Enhanced DADA Model for Polyps Detection.pptx]
- **Report:** [Enhanced DADA Model.pdf]

### Reference Paper
- [Targeted False Positive Synthesis via Detector-guided Adversarial Diffusion Attacker for Robust Polyp Detection](https://arxiv.org/abs/2506.18134)

### Reference Dataset
- [Kvasir Dataset](https://datasets.simula.no/kvasir/)
- [ETIS-Larib Polyp DB Dataset] (https://service.tib.eu/ldmservice/dataset/etis-larib-polyp-db)


## Project Technicalities

### Terminologies
- **Diffusion Model:** A generative model that progressively transforms random noise into coherent data.
- **UNet Architecture:** A neural network with an encoder-decoder structure featuring skip connections for better feature preservation.
- **Perceptual Loss:** A loss function that measures high-level differences between images, emphasizing perceptual similarity.

### Problem Statements
- **Problem 1:** Achieving high-resolution and detailed images using conventional diffusion models remains challenging.
- **Problem 2:** Existing models suffer from slow inference times during the image generation process.

### Loopholes or Research Areas
- **Evaluation Metrics:** Lack of robust metrics to effectively assess the quality of generated images.
- **Output Consistency:** Inconsistencies in output quality when scaling the model to higher resolutions.
- **Computational Resources:** Training requires significant GPU compute resources, which may not be readily accessible.

### Problem vs. Ideation: Proposed 3 Ideas to Solve the Problems
1. **Optimized Architecture:** Redesign the model architecture to improve efficiency and balance image quality with faster inference.
2. **Advanced Loss Functions:** Integrate novel loss functions (e.g., perceptual loss) to better capture artistic nuances and structural details.
3. **Enhanced Data Augmentation:** Implement sophisticated data augmentation strategies to improve the model’s robustness and reduce overfitting.

### Proposed Solution: Code-Based Implementation
This repository provides an implementation of the enhanced stable diffusion model using PyTorch. The solution includes:

- **Modified UNet Architecture:** Incorporates residual connections and efficient convolutional blocks.
- **Novel Loss Functions:** Combines Mean Squared Error (MSE) with perceptual loss to enhance feature learning.
- **Optimized Training Loop:** Reduces computational overhead while maintaining performance.

### Key Components
- **`model.py`**: Contains the modified UNet architecture and other model components.
- **`train.py`**: Script to handle the training and evaluation process 
- **`Main.py`**: configurable parameters.
- **`loss.py`**: loss functions
- **`metrics.py`**:  metric evaluations.

## Model Workflow
The workflow of the Enhanced Stable Diffusion model is designed to generate  high-quality false-positive synthetic images.

1. **Input:**
   - Base Image: A real colonoscopy frame.
   - Mask: The target region to inpaint (simulating a false positive).
   - Noise Tensor: Random noise initiating the diffusion trajectory.

2. **Diffusion Process:**
   - The UNet-based DDPM performs iterative denoising. At each step:
   - YOLO provides adversarial gradients → pushes the sample toward false-positive features.
   - Background VGG perceptual + style loss keeps anatomical realism.
   - Region-adaptive α modifies attack strength per region.

3. **Output:**
   - A final synthetic false-positive image
   - Realistic background, polyp-like artifact
   - Used to augment polyp detection datasets

## How to Run the Code

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/BRAIN-Lab-AI/Enhancing-Targeted-False-Positive-Synthesis-via-Detector-guided-Adversarial-Diffusion-Attacker.git
    cd Enhancing-Targeted-False-Positive-Synthesis-via-Detector-guided-Adversarial-Diffusion-Attacker.git
    ```

2. **Set Up the Environment:**
    Create a virtual environment and install the required dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    pip install -r requirements.txt
    ```
**Data synthesis**
We provide well-trained BG-De weights based on the public Kvasir dataset, which can be downloaded from this [link](https://drive.google.com/file/d/18_8oLJduhYCx7lbAsfmh6HbS4ZEkQY9C/view), . Please place the weights in the BG-De_model folder.

Additionally, we also offer YOLOv5l weights trained on the public Kvasir dataset, which can be downloaded from this [link](https://drive.google.com/file/d/1hfs5trwjaZXrCVflEZstlHSiioMQJEp4/view). Please place the weights in the Detection_model folder.

After completing the above steps, you can generate negative samples by simply running Main.py:
```bash
    python Main.py
    ```


3. **Train the Model:**

** Prepare the datasets for BG-De.**
Please place the dataset you want to train in the path ./datasets and ensure that the size of the images and masks is 256. The path structure should be as follows:
  DADA
  ├── datasets
  │   ├── images
  │   ├── masks

Train your own BG-De.
Please set the "state" parameter of modelConfig in Main.py to "train", and set parameters such as batch_size according to actual conditions.

    ```bash
    python Main.py
    ```
Place the weights in the BG-De_model folder.
4. **Testing:**

Before running the tests, ensure an object detection model is ready. For this project, we utilize the YOLOv5l architecture. Place the weights in the Detection_model folder and set the "state" parameter of modelConfig in Main.py to "eval".

    ```bash
    python Main.py 
    ```

## Acknowledgments
- **Open-Source Communities:** Thanks to the contributors of PyTorch, Hugging Face, and other libraries for their amazing work.
- **Individuals:** Special thanks to Dr. Muzammil Behzadand for this experience, and I acknowledge the outstanding open-source contributions from [DADA](https://github.com/Huster-Hq/DADA/tree/main), [DDPM](https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-), [YOLOv5](https://github.com/ultralytics/yolov5), and [DETR](https://github.com/facebookresearch/detr).

