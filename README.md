# Neural Style Transfer with PyTorch

This repository contains a PyTorch implementation of neural style transfer, allowing you to apply the artistic style of one image to another.

## AI and Deep Learning Implementation

This project utilizes AI and deep learning through neural style transfer, which combines the content of one image with the artistic style of another. Here's how it works:

1. **Deep Neural Network Architecture**:
   - Uses VGG19, a pre-trained convolutional neural network (CNN) originally trained on ImageNet
   - VGG19 serves as a feature extractor to understand both content and style of images

2. **Content and Style Representation**:
   - **Content Representation**: Uses higher layers of VGG19 (specifically 'conv_4_2') to capture image content
   - **Style Representation**: Uses multiple layers ('conv_1_1' through 'conv_5_1') to capture style through Gram matrices
   - Gram matrices represent correlations between different features, capturing the artistic style

3. **Optimization Process**:
   - Uses gradient descent optimization (LBFGS optimizer)
   - Minimizes a loss function with two components:
     - Content loss: Measures content similarity with original image
     - Style loss: Measures style similarity with style image
   - Iteratively updates the generated image to minimize these losses

4. **Key AI/Deep Learning Components**:
   - Feature extraction using CNN layers
   - Gram matrix computation for style representation
   - Loss function optimization
   - Backpropagation for gradient computation
   - Transfer learning (using pre-trained VGG19)

This implementation is based on the original neural style transfer paper by Gatys et al. (2016), demonstrating how deep learning can create artistic transformations by separating and recombining content and style representations.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

To perform style transfer, you'll need two images:
- A content image (the image you want to transform)
- A style image (the image whose style you want to apply)

Run the style transfer using:
```bash
python run_style_transfer.py --content path/to/content.jpg --style path/to/style.jpg
```

### Additional Options

- `--output_dir`: Directory to save results (default: 'output')
- `--style_weight`: Weight for style loss (default: 1e5)
- `--content_weight`: Weight for content loss (default: 1)
- `--iterations`: Number of optimization iterations (default: 500)
- `--max_dim`: Maximum dimension of input images (default: 512)

Example with custom parameters:
```bash
python run_style_transfer.py --content content.jpg --style style.jpg --style_weight 1e6 --content_weight 10 --iterations 1000
```

## Output

The script will create an output directory containing:
- The original content image
- The style image
- The stylized result
- A combined image showing all three images side by side

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- NumPy
- Pillow
- matplotlib

## Notes

- The style transfer process may take several minutes depending on your hardware and the number of iterations
- For best results, use high-quality images
- Adjust the style_weight and content_weight parameters to achieve different artistic effects 