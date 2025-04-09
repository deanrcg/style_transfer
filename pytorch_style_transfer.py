import os
import argparse
import time
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
from datetime import datetime

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

class StyleTransfer:
    def __init__(self, device=None):
        """Initialize the style transfer model"""
        # Determine device (CPU/GPU)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Load pre-trained VGG19 model
        self.model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(self.device).eval()
        
        # Define content and style layers
        self.content_layers = ['conv_4_2']
        self.style_layers = ['conv_1_1', 'conv_2_1', 'conv_3_1', 'conv_4_1', 'conv_5_1']
        
        # VGG normalization constants
        self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.norm_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)

    def load_img(self, path_to_img, max_dim=512):
        """Load and preprocess an image for the model"""
        # Check if file exists
        if not os.path.exists(path_to_img):
            raise FileNotFoundError(f"Image file not found: {path_to_img}")
            
        # Load the image with PIL
        img = PIL.Image.open(path_to_img).convert('RGB')
        
        # Resize image while preserving aspect ratio
        if max(img.size) > max_dim:
            scale = max_dim / max(img.size)
            size = tuple([int(x * scale) for x in img.size])
            img = img.resize(size, PIL.Image.LANCZOS)
            
        # Convert to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        img = transform(img)
        img = img.unsqueeze(0).to(self.device)
        
        return img

    def tensor_to_image(self, tensor):
        """Convert a tensor to a PIL Image"""
        tensor = tensor.cpu().clone().detach()
        tensor = tensor.squeeze(0)
        tensor = tensor.div(255.0)
        
        # Convert to PIL image
        transform = transforms.ToPILImage()
        return transform(tensor)

    def gram_matrix(self, input_tensor):
        """Calculate Gram matrix for style representation"""
        batch_size, channels, height, width = input_tensor.size()
        features = input_tensor.view(batch_size, channels, height * width)
        features_t = features.transpose(1, 2)
        
        gram = features.bmm(features_t)
        return gram.div(channels * height * width)

    def get_vgg_layers(self):
        """Return a dictionary mapping layer names to their indices in VGG19"""
        vgg_layers = {
            'conv_1_1': 0, 'conv_1_2': 2,
            'conv_2_1': 5, 'conv_2_2': 7,
            'conv_3_1': 10, 'conv_3_2': 12, 'conv_3_3': 14, 'conv_3_4': 16,
            'conv_4_1': 19, 'conv_4_2': 21, 'conv_4_3': 23, 'conv_4_4': 25,
            'conv_5_1': 28, 'conv_5_2': 30, 'conv_5_3': 32, 'conv_5_4': 34
        }
        return vgg_layers

    def get_features(self, image, layers):
        """Extract features from specified layers"""
        features = {}
        x = image
        
        # Normalize the image
        x = x.clone().div(255.0)
        x = x.sub(self.norm_mean.view(1, -1, 1, 1))
        x = x.div(self.norm_std.view(1, -1, 1, 1))
            
        # Get features from each layer
        for i, (name, module) in enumerate(self.model._modules.items()):
            x = module(x)
            layer_name = self.get_layer_name(i)
            if layer_name in layers:
                features[layer_name] = x
                
        return features

    def get_layer_name(self, index):
        """Convert layer index to layer name"""
        layer_mapping = {
            0: 'conv_1_1', 2: 'conv_1_2',
            5: 'conv_2_1', 7: 'conv_2_2',
            10: 'conv_3_1', 12: 'conv_3_2', 14: 'conv_3_3', 16: 'conv_3_4',
            19: 'conv_4_1', 21: 'conv_4_2', 23: 'conv_4_3', 25: 'conv_4_4',
            28: 'conv_5_1', 30: 'conv_5_2', 32: 'conv_5_3', 34: 'conv_5_4'
        }
        return layer_mapping.get(index, None)

    def transfer_style(self, content_image, style_image, style_weight=1e5, 
                       content_weight=1, iterations=500, show_progress=True):
        """Perform style transfer between content and style images"""
        print(f"Extracting features from content and style images...")
        
        # Create a white noise image to optimize
        input_img = content_image.clone()
        
        # Get content and style features
        content_features = self.get_features(content_image, self.content_layers)
        style_features = self.get_features(style_image, self.style_layers)
        
        # Calculate gram matrices for style features
        style_grams = {layer: self.gram_matrix(style_features[layer]) for layer in style_features}
        
        # Create optimizer
        input_img.requires_grad_(True)
        optimizer = optim.LBFGS([input_img])
        
        # Create progress bar if requested
        if show_progress:
            print(f"\nStarting style transfer process with {iterations} iterations")
            print(f"Style weight: {style_weight}, Content weight: {content_weight}")
            print(f"{'Iteration':<10}{'Loss':<15}{'Time':<10}")
            print("-" * 35)
        
        start_time = time.time()
        best_image = None
        best_loss = float('inf')
        
        # Style transfer optimization loop
        run = [0]
        while run[0] <= iterations:
            iteration_start = time.time()
            
            def closure():
                # Zero the gradients
                optimizer.zero_grad()
                
                # Get features of current image
                features = self.get_features(input_img, self.content_layers + self.style_layers)
                
                # Calculate content loss
                content_loss = 0
                for layer in self.content_layers:
                    target = content_features[layer].detach()
                    content_loss += F.mse_loss(features[layer], target)
                content_loss *= content_weight
                
                # Calculate style loss
                style_loss = 0
                for layer in self.style_layers:
                    current = features[layer]
                    gram = self.gram_matrix(current)
                    target = style_grams[layer].detach()
                    style_loss += F.mse_loss(gram, target)
                style_loss *= style_weight / len(self.style_layers)
                
                # Total loss
                total_loss = content_loss + style_loss
                
                # Calculate gradients
                total_loss.backward()
                
                # Track loss
                run[0] += 1
                if run[0] % 50 == 0 or run[0] == iterations:
                    if show_progress:
                        iter_time = time.time() - iteration_start
                        print(f"{run[0]:<10}{total_loss.item():<15.4f}{iter_time:<10.4f}s")
                
                # Keep track of best image
                nonlocal best_loss, best_image
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    best_image = input_img.clone()
                
                return total_loss
            
            # Perform optimization step
            optimizer.step(closure)
            
            # Ensure image values stay in valid range
            with torch.no_grad():
                input_img.clamp_(0, 255)
                
            # Stop if reached max iterations
            if run[0] >= iterations:
                break
        
        # Return best image
        total_time = time.time() - start_time
        print(f"\nStyle transfer completed in {total_time:.2f} seconds")
        
        if best_image is not None:
            return best_image
        return input_img

    def save_images(self, content_image, style_image, stylized_image, output_dir="output"):
        """Save all images to output directory with timestamp"""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert images to PIL format
        content_pil = self.tensor_to_image(content_image)
        style_pil = self.tensor_to_image(style_image)
        stylized_pil = self.tensor_to_image(stylized_image)
        
        # Define filenames
        content_filename = os.path.join(output_dir, f"content_{timestamp}.jpg")
        style_filename = os.path.join(output_dir, f"style_{timestamp}.jpg")
        result_filename = os.path.join(output_dir, f"stylized_{timestamp}.jpg")
        
        # Save images
        content_pil.save(content_filename)
        style_pil.save(style_filename)
        stylized_pil.save(result_filename)
        
        # Save combined image for comparison
        width = max(content_pil.width, style_pil.width, stylized_pil.width)
        height = content_pil.height + style_pil.height + stylized_pil.height
        
        combined = PIL.Image.new('RGB', (width, height))
        combined.paste(content_pil, (0, 0))
        combined.paste(style_pil, (0, content_pil.height))
        combined.paste(stylized_pil, (0, content_pil.height + style_pil.height))
        
        combined_filename = os.path.join(output_dir, f"combined_{timestamp}.jpg")
        combined.save(combined_filename)
        
        return {
            'content': content_filename,
            'style': style_filename,
            'result': result_filename,
            'combined': combined_filename
        }

    def display_images(self, content_image, style_image, stylized_image):
        """Display images using matplotlib"""
        plt.figure(figsize=(12, 12))
        
        content = self.tensor_to_image(content_image)
        style = self.tensor_to_image(style_image)
        stylized = self.tensor_to_image(stylized_image)
        
        plt.subplot(1, 3, 1)
        plt.imshow(content)
        plt.title('Content Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(style)
        plt.title('Style Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(stylized)
        plt.title('Stylized Image')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Neural Style Transfer with PyTorch')
    parser.add_argument('content_image', type=str, help='Path to content image')
    parser.add_argument('style_image', type=str, help='Path to style image')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--iterations', type=int, default=500, help='Number of iterations')
    parser.add_argument('--content-weight', type=float, default=1, help='Content weight')
    parser.add_argument('--style-weight', type=float, default=1e5, help='Style weight')
    parser.add_argument('--max-dim', type=int, default=512, help='Maximum image dimension')
    parser.add_argument('--no-display', action='store_true', help='Do not display images')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    
    args = parser.parse_args()
    
    try:
        # Initialize style transfer
        device = torch.device('cpu') if args.cpu else None
        transferrer = StyleTransfer(device=device)
        
        # Load images
        print(f"Loading content image: {args.content_image}")
        content_image = transferrer.load_img(args.content_image, max_dim=args.max_dim)
        
        print(f"Loading style image: {args.style_image}")
        style_image = transferrer.load_img(args.style_image, max_dim=args.max_dim)
        
        # Perform style transfer
        stylized_image = transferrer.transfer_style(
            content_image, 
            style_image,
            style_weight=args.style_weight,
            content_weight=args.content_weight,
            iterations=args.iterations
        )
        
        # Save images
        output_files = transferrer.save_images(
            content_image, 
            style_image, 
            stylized_image, 
            output_dir=args.output_dir
        )
        
        print(f"\nImages saved:")
        for key, path in output_files.items():
            print(f"- {key.capitalize()}: {path}")
        
        # Display images
        if not args.no_display:
            transferrer.display_images(content_image, style_image, stylized_image)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    # Run main function
    exit(main())