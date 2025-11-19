![ViT](https://github.com/user-attachments/assets/73764837-a38b-46ce-a043-95bee2e81a65)# Targeted False Positive Synthesis via Detector-guided Adversarial Diffusion Attacker for Robust Polyp Detection

## Project Metadata
### Authors
- **Team:** Dalal Aldowaihi
- **Supervisor Name:** Dr. Muzammil Behzad
- **Affiliations:** IAU and KFUPM 

## Introduction
Deep learning models, particularly Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs), have achieved remarkable success in medical image classification. However, despite their impressive performance, these models remain highly vulnerable to adversarial attacks, small imperceptible perturbations that can cause misclassification. Such subtle modifications to input images can mislead a model’s predictions, raising concerns about the robustness, reliability, and trustworthiness of AI-driven diagnostic systems.

To address this challenge, researchers have proposed methods to improve adversarial robustness without significantly compromising classification accuracy. A recent approach, Self-Ensembling Vision Transformer (SEViT), leverages intermediate block features within a ViT to mitigate the impact of adversarial perturbations and enhance robustness in medical imaging tasks. By training multiple classifiers on early transformer block outputs and combining their predictions with the final classifier, SEViT improves resilience and enables effective adversarial sample detection.


## Problem Statement
Polyp detection models are often sensitive to complex colon backgrounds—such as folds, specular highlights, shadows, and circular lumen structures, which can resemble real polyps.  
This causes **high false-positive rates**, leading to:
- Unnecessary alerts  
- Interrupted clinical workflow  
- Increased cognitive load for doctors  
- Reduced trust in AI systems  

Existing synthetic data techniques mainly generate **positive samples** (polyp images), but **almost none** produce *challenging negative samples* that intentionally mimic false positives.


## Application Area and Project Domain
**Application Area:** Medical Image classification with robustness against adversarial attacks, with specific focus on disease classification tasks (e.g., tuberculosis detection from chest X-rays, diabetic retinopathy detection from retinal images).

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

  
# THE FOLLOWING IS SUPPOSED TO BE DONE LATER

### Project Documents
- **Presentation:** [Project Presentation](/presentation.pptx)
- **Report:** [Project Report](/report.pdf)

### Reference Paper
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

### Reference Dataset
- [LAION-5B Dataset](https://laion.ai/blog/laion-5b/)


## Project Technicalities

### Terminologies
- **Diffusion Model:** A generative model that progressively transforms random noise into coherent data.
- **Latent Space:** A compressed, abstract representation of data where complex features are captured.
- **UNet Architecture:** A neural network with an encoder-decoder structure featuring skip connections for better feature preservation.
- **Text Encoder:** A model that converts text into numerical embeddings for downstream tasks.
- **Perceptual Loss:** A loss function that measures high-level differences between images, emphasizing perceptual similarity.
- **Tokenization:** The process of breaking down text into smaller units (tokens) for processing.
- **Noise Vector:** A randomly generated vector used to initialize the diffusion process in generative models.
- **Decoder:** A network component that transforms latent representations back into image space.
- **Iterative Refinement:** The process of gradually improving the quality of generated data through multiple steps.
- **Conditional Generation:** The process where outputs are generated based on auxiliary inputs, such as textual descriptions.

### Problem Statements
- **Problem 1:** Achieving high-resolution and detailed images using conventional diffusion models remains challenging.
- **Problem 2:** Existing models suffer from slow inference times during the image generation process.
- **Problem 3:** There is limited capability in performing style transfer and generating diverse artistic variations.

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
- **`train.py`**: Script to handle the training process with configurable parameters.
- **`utils.py`**: Utility functions for data processing, augmentation, and metric evaluations.
- **`inference.py`**: Script for generating images using the trained model.

## Model Workflow
The workflow of the Enhanced Stable Diffusion model is designed to translate textual descriptions into high-quality artistic images through a multi-step diffusion process:

1. **Input:**
   - **Text Prompt:** The model takes a text prompt (e.g., "A surreal landscape with mountains and rivers") as the primary input.
   - **Tokenization:** The text prompt is tokenized and processed through a text encoder (such as a CLIP model) to obtain meaningful embeddings.
   - **Latent Noise:** A random latent noise vector is generated to initialize the diffusion process, which is then conditioned on the text embeddings.

2. **Diffusion Process:**
   - **Iterative Refinement:** The conditioned latent vector is fed into a modified UNet architecture. The model iteratively refines this vector by reversing a diffusion process, gradually reducing noise while preserving the text-conditioned features.
   - **Intermediate States:** At each step, intermediate latent representations are produced that increasingly capture the structure and details dictated by the text prompt.

3. **Output:**
   - **Decoding:** The final refined latent representation is passed through a decoder (often part of a Variational Autoencoder setup) to generate the final image.
   - **Generated Image:** The output is a synthesized image that visually represents the input text prompt, complete with artistic style and detail.

## How to Run the Code

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/enhanced-stable-diffusion.git
    cd enhanced-stable-diffusion
    ```

2. **Set Up the Environment:**
    Create a virtual environment and install the required dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Train the Model:**
    Configure the training parameters in the provided configuration file and run:
    ```bash
    python train.py --config configs/train_config.yaml
    ```

4. **Generate Images:**
    Once training is complete, use the inference script to generate images.
    ```bash
    python inference.py --checkpoint path/to/checkpoint.pt --input "A surreal landscape with mountains and rivers"
    ```

## Acknowledgments
- **Open-Source Communities:** Thanks to the contributors of PyTorch, Hugging Face, and other libraries for their amazing work.
- **Individuals:** Special thanks to bla, bla, bla for the amazing team effort, invaluable guidance and support throughout this project.
- **Resource Providers:** Gratitude to ABC-organization for providing the computational resources necessary for this project.
