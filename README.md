# Neural Style Transfer with PyTorch

This repository contains a PyTorch implementation of neural style transfer, allowing you to apply the artistic style of one image to another.

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