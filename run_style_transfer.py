import argparse
from pytorch_style_transfer import StyleTransfer

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Neural Style Transfer')
    parser.add_argument('--content', type=str, required=True, help='Path to content image')
    parser.add_argument('--style', type=str, required=True, help='Path to style image')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save results')
    parser.add_argument('--style_weight', type=float, default=1e5, help='Weight for style loss')
    parser.add_argument('--content_weight', type=float, default=1, help='Weight for content loss')
    parser.add_argument('--iterations', type=int, default=500, help='Number of optimization iterations')
    parser.add_argument('--max_dim', type=int, default=512, help='Maximum dimension of input images')
    
    args = parser.parse_args()
    
    # Initialize style transfer
    style_transfer = StyleTransfer()
    
    try:
        # Load images
        print("Loading images...")
        content_image = style_transfer.load_img(args.content, max_dim=args.max_dim)
        style_image = style_transfer.load_img(args.style, max_dim=args.max_dim)
        
        # Perform style transfer
        print("Starting style transfer...")
        stylized_image = style_transfer.transfer_style(
            content_image=content_image,
            style_image=style_image,
            style_weight=args.style_weight,
            content_weight=args.content_weight,
            iterations=args.iterations
        )
        
        # Save results
        print("Saving results...")
        style_transfer.save_images(
            content_image=content_image,
            style_image=style_image,
            stylized_image=stylized_image,
            output_dir=args.output_dir
        )
        
        print("Style transfer completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 