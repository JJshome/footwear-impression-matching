"""
Interactive Demo for Footwear Impression Matching

This script provides a user-friendly demo interface for the footwear impression matching system.
It allows users to:
1. Load a trained model
2. Select a track (crime scene) impression
3. Match it against a reference database
4. Visualize the results

Usage:
    python demo.py --model path/to/model.pth --ref_dir path/to/references
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2

# Import project modules
from data.dataloader import get_transforms
from models.network import FootwearMatchingNetwork, EnhancedSiameseNetwork
from utils.common import set_seed


class FootwearMatchingDemo:
    """
    Interactive demo for footwear impression matching.
    """
    
    def __init__(self, model_path, ref_dir, backbone='resnet50', feature_dim=256, img_size=512):
        """
        Initialize the demo.
        
        Args:
            model_path: Path to trained model
            ref_dir: Directory with reference images
            backbone: Backbone architecture
            feature_dim: Feature dimension
            img_size: Image size
        """
        self.model_path = model_path
        self.ref_dir = ref_dir
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.img_size = img_size
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Set random seed
        set_seed(42)
        
        # Load model
        self.load_model()
        
        # Setup transforms
        self.transform = get_transforms(mode='val', img_size=img_size)
        
        # Load reference database
        self.load_references()
        
        # GUI components
        self.root = None
        self.track_img_display = None
        self.track_path = None
        self.results_canvas = None
        self.fig = None
        self.ax = None
        
        # Initialize GUI
        self.setup_gui()
    
    def load_model(self):
        """Load the trained model."""
        print("Loading model...")
        
        try:
            # Create model
            self.model = FootwearMatchingNetwork(
                backbone=self.backbone,
                pretrained=False,
                feature_dim=self.feature_dim
            )
            
            # Load weights
            state_dict = torch.load(self.model_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print("Model loaded successfully!")
        
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise e
    
    def load_references(self):
        """Load reference database."""
        print("Loading reference database...")
        
        self.ref_paths = []
        self.ref_ids = []
        
        try:
            for filename in os.listdir(self.ref_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    ref_path = os.path.join(self.ref_dir, filename)
                    ref_id = os.path.splitext(filename)[0]
                    self.ref_paths.append(ref_path)
                    self.ref_ids.append(ref_id)
            
            print(f"Loaded {len(self.ref_paths)} reference images")
        
        except Exception as e:
            print(f"Error loading references: {str(e)}")
            raise e
    
    def setup_gui(self):
        """Setup the GUI."""
        # Create main window
        self.root = tk.Tk()
        self.root.title("Footwear Impression Matching Demo")
        self.root.geometry("1200x800")
        
        # Title label
        title_label = tk.Label(self.root, text="Footwear Impression Matching", font=("Arial", 18, "bold"))
        title_label.pack(pady=10)
        
        # Frame for controls
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Load Track Image button
        load_button = tk.Button(control_frame, text="Load Track Image", command=self.load_track_image, width=20)
        load_button.pack(side=tk.LEFT, padx=5)
        
        # Match button
        match_button = tk.Button(control_frame, text="Match", command=self.match_impression, width=20)
        match_button.pack(side=tk.LEFT, padx=5)
        
        # Top-k slider
        self.top_k_var = tk.IntVar(value=5)
        tk.Label(control_frame, text="Top matches:").pack(side=tk.LEFT, padx=(20, 5))
        top_k_slider = tk.Scale(control_frame, from_=1, to=10, orient=tk.HORIZONTAL, variable=self.top_k_var)
        top_k_slider.pack(side=tk.LEFT)
        
        # Frame for track image
        track_frame = tk.Frame(self.root)
        track_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Label for track image
        tk.Label(track_frame, text="Track (Crime Scene) Impression", font=("Arial", 12)).pack()
        
        # Canvas for track image
        self.track_img_display = tk.Label(track_frame)
        self.track_img_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Frame for results
        results_frame = tk.Frame(self.root)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Label for results
        tk.Label(results_frame, text="Matching Results", font=("Arial", 12)).pack()
        
        # Initialize matplotlib figure for results
        self.fig, self.ax = plt.subplots(1, 3, figsize=(12, 4))
        self.fig.tight_layout()
        
        # Canvas for results
        self.results_canvas = FigureCanvasTkAgg(self.fig, master=results_frame)
        self.results_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_track_image(self):
        """Load a track image from file."""
        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select Track Image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg")]
        )
        
        if not file_path:
            return
        
        try:
            # Load and display image
            self.track_path = file_path
            self.status_var.set(f"Loaded track image: {os.path.basename(file_path)}")
            
            # Display image
            img = Image.open(file_path)
            img = img.resize((384, 384), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            
            self.track_img_display.config(image=img_tk)
            self.track_img_display.image = img_tk  # Keep a reference
            
            # Clear results
            for a in self.ax:
                a.clear()
            self.results_canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            self.status_var.set("Error loading image")
    
    def match_impression(self):
        """Match the loaded track impression against the reference database."""
        if not self.track_path:
            messagebox.showwarning("Warning", "Please load a track image first")
            return
        
        try:
            self.status_var.set("Matching impression...")
            
            # Load track image
            track_img = Image.open(self.track_path).convert('RGB')
            track_tensor = self.transform(track_img).unsqueeze(0).to(self.device)
            
            # Process in batches
            batch_size = 32
            similarities = []
            
            with torch.no_grad():
                # Extract track features once
                track_features = self.model(track_tensor, None, mode='track')
                
                # Process reference images in batches
                for i in range(0, len(self.ref_paths), batch_size):
                    batch_paths = self.ref_paths[i:i+batch_size]
                    
                    # Load and preprocess reference images
                    ref_tensors = []
                    for ref_path in batch_paths:
                        ref_img = Image.open(ref_path).convert('RGB')
                        ref_tensor = self.transform(ref_img)
                        ref_tensors.append(ref_tensor)
                    
                    # Stack tensors and move to device
                    ref_batch = torch.stack(ref_tensors).to(self.device)
                    
                    # Extract reference features
                    ref_features = self.model(None, ref_batch, mode='ref')
                    
                    # Compute similarities
                    for j in range(len(batch_paths)):
                        # Extract single reference features
                        single_ref_features = ref_features[j:j+1]
                        
                        # Compute similarity
                        similarity = self.model.compute_mcncc(track_features, single_ref_features)
                        
                        # Store result
                        similarities.append((self.ref_ids[i+j], self.ref_paths[i+j], similarity.item()))
            
            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[2], reverse=True)
            
            # Display top matches
            top_k = self.top_k_var.get()
            top_matches = similarities[:top_k]
            
            self.display_results(track_img, top_matches)
            
            self.status_var.set(f"Matched against {len(self.ref_paths)} references. Displaying top {top_k} matches.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error during matching: {str(e)}")
            self.status_var.set("Error during matching")
    
    def display_results(self, track_img, matches):
        """
        Display matching results.
        
        Args:
            track_img: Track image
            matches: List of (ref_id, ref_path, score) tuples
        """
        # Clear axes
        for a in self.ax:
            a.clear()
        
        # Display track image
        self.ax[0].imshow(track_img)
        self.ax[0].set_title("Query Track Image")
        self.ax[0].axis('off')
        
        # Display top match
        if matches:
            top_id, top_path, top_score = matches[0]
            top_img = Image.open(top_path)
            self.ax[1].imshow(top_img)
            self.ax[1].set_title(f"Best Match\nID: {top_id}\nScore: {top_score:.4f}")
            self.ax[1].axis('off')
        
        # Display all scores
        y_pos = range(len(matches))
        scores = [score for _, _, score in matches]
        ids = [ref_id for ref_id, _, _ in matches]
        
        bars = self.ax[2].barh(y_pos, scores, align='center')
        self.ax[2].set_yticks(y_pos)
        self.ax[2].set_yticklabels(ids)
        self.ax[2].invert_yaxis()  # Labels read top-to-bottom
        self.ax[2].set_xlabel('Similarity Score')
        self.ax[2].set_title('Top Matches')
        
        # Add score labels
        for i, bar in enumerate(bars):
            self.ax[2].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f"{scores[i]:.4f}", va='center')
        
        self.fig.tight_layout()
        self.results_canvas.draw()
    
    def run(self):
        """Run the application."""
        try:
            # Import ImageTk here to avoid issues if not available
            global ImageTk
            from PIL import ImageTk
            
            self.root.mainloop()
        except ImportError:
            print("Error: Tkinter or PIL.ImageTk not available.")
            print("Running in command-line mode instead.")
            self.run_cli()
    
    def run_cli(self):
        """Run in command-line mode (if GUI is not available)."""
        print("\nRunning in command-line mode")
        print("============================")
        
        while True:
            # Get track image path
            track_path = input("\nEnter path to track image (or 'q' to quit): ")
            
            if track_path.lower() == 'q':
                break
            
            if not os.path.exists(track_path):
                print(f"Error: File {track_path} does not exist")
                continue
            
            # Set track path
            self.track_path = track_path
            
            try:
                # Load track image
                track_img = Image.open(self.track_path).convert('RGB')
                track_tensor = self.transform(track_img).unsqueeze(0).to(self.device)
                
                print(f"Matching against {len(self.ref_paths)} references...")
                
                # Process in batches
                batch_size = 32
                similarities = []
                
                with torch.no_grad():
                    # Extract track features once
                    track_features = self.model(track_tensor, None, mode='track')
                    
                    # Process reference images in batches
                    for i in range(0, len(self.ref_paths), batch_size):
                        batch_paths = self.ref_paths[i:i+batch_size]
                        
                        # Load and preprocess reference images
                        ref_tensors = []
                        for ref_path in batch_paths:
                            ref_img = Image.open(ref_path).convert('RGB')
                            ref_tensor = self.transform(ref_img)
                            ref_tensors.append(ref_tensor)
                        
                        # Stack tensors and move to device
                        ref_batch = torch.stack(ref_tensors).to(self.device)
                        
                        # Extract reference features
                        ref_features = self.model(None, ref_batch, mode='ref')
                        
                        # Compute similarities
                        for j in range(len(batch_paths)):
                            # Extract single reference features
                            single_ref_features = ref_features[j:j+1]
                            
                            # Compute similarity
                            similarity = self.model.compute_mcncc(track_features, single_ref_features)
                            
                            # Store result
                            similarities.append((self.ref_ids[i+j], similarity.item()))
                
                # Sort by similarity (highest first)
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                # Get top matches
                top_k = 5
                top_matches = similarities[:top_k]
                
                print("\nTop Matches:")
                for i, (ref_id, score) in enumerate(top_matches):
                    print(f"{i+1}. Reference ID: {ref_id}, Similarity: {score:.4f}")
                
                # Save visualization
                self.save_cli_visualization(track_img, top_matches)
                
            except Exception as e:
                print(f"Error during matching: {str(e)}")
    
    def save_cli_visualization(self, track_img, matches):
        """
        Save visualization in command-line mode.
        
        Args:
            track_img: Track image
            matches: List of (ref_id, score) tuples
        """
        output_dir = "demo_results"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_path = os.path.join(output_dir, f"matches_{timestamp}.png")
        
        n = len(matches)
        plt.figure(figsize=(15, 4 * (n+1)))
        plt.suptitle("Track Image and Top Matches", fontsize=16)
        
        # Display track image
        plt.subplot(n+1, 2, 1)
        plt.imshow(track_img)
        plt.title("Query Track Image")
        plt.axis('off')
        
        # Display info
        plt.subplot(n+1, 2, 2)
        plt.text(0.5, 0.5, 
                f"File: {os.path.basename(self.track_path)}\n"
                f"Top {n} matches shown",
                ha='center', va='center', fontsize=12)
        plt.axis('off')
        
        # Display matches
        for i, (ref_id, score) in enumerate(matches):
            # Find reference image path
            ref_path = None
            for path, id in zip(self.ref_paths, self.ref_ids):
                if id == ref_id:
                    ref_path = path
                    break
            
            if ref_path:
                ref_img = Image.open(ref_path).convert('RGB')
                
                plt.subplot(n+1, 2, 3 + i*2)
                plt.imshow(ref_img)
                plt.title(f"Match #{i+1}: Reference {ref_id}")
                plt.axis('off')
                
                plt.subplot(n+1, 2, 4 + i*2)
                plt.text(0.5, 0.5, 
                        f"Similarity Score: {score:.4f}",
                        ha='center', va='center', fontsize=12)
                plt.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(output_path)
        plt.close()
        print(f"Visualization saved to {output_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Footwear Impression Matching Demo')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--ref_dir', type=str, required=True, help='Directory with reference images')
    parser.add_argument('--backbone', type=str, default='resnet50', help='Backbone model')
    parser.add_argument('--feature_dim', type=int, default=256, help='Feature dimension')
    parser.add_argument('--img_size', type=int, default=512, help='Image size')
    return parser.parse_args()


if __name__ == '__main__':
    import time
    
    # Parse arguments
    args = parse_args()
    
    # Create and run demo
    demo = FootwearMatchingDemo(
        args.model,
        args.ref_dir,
        args.backbone,
        args.feature_dim,
        args.img_size
    )
    
    demo.run()
