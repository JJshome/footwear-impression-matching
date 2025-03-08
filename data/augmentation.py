"""
Data Augmentation Module for Footwear Impressions

This module implements specialized augmentation techniques for footwear impressions,
ensuring that augmented images maintain the white background outside the impression area.

Key augmentations:
1. Rotation and affine transformations
2. Elastic deformations
3. Intensity adjustments
4. Noise addition
5. Perspective transformations
6. Simulated impression artifacts (partial impressions, smudges)

Usage:
    from data.augmentation import FootwearAugmenter
    augmenter = FootwearAugmenter()
    augmented_img = augmenter.augment(img)
"""

import cv2
import numpy as np
import random
from PIL import Image, ImageEnhance, ImageFilter
from scipy.ndimage import gaussian_filter, map_coordinates

class FootwearAugmenter:
    """
    Augmentation class for footwear impression images.
    Specializes in maintaining white backgrounds and realistic impression distortions.
    """
    
    def __init__(self, seed=None):
        """
        Initialize the augmenter.
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Define augmentation probabilities
        self.aug_probs = {
            'rotate': 0.5,
            'brightness': 0.4,
            'contrast': 0.4,
            'noise': 0.3,
            'elastic': 0.3,
            'perspective': 0.3,
            'cutout': 0.25,
            'smudge': 0.3,
            'artifacts': 0.25,
            'random_lines': 0.2
        }
    
    def _create_impression_mask(self, img, threshold=240):
        """
        Create a mask of the impression area (non-white regions).
        
        Args:
            img: Input image
            threshold: Threshold for white pixels
            
        Returns:
            Binary mask of impression area (1 for impression, 0 for background)
        """
        # Convert to grayscale if not already
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Threshold to create a binary mask
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Dilate slightly to ensure coverage
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Normalize to 0-1 range
        mask = mask / 255.0
        
        return mask
    
    def rotate_image(self, img, max_angle=30):
        """
        Rotate the image while maintaining white background.
        
        Args:
            img: Input image
            max_angle: Maximum rotation angle
            
        Returns:
            Rotated image
        """
        angle = random.uniform(-max_angle, max_angle)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        # Compute the rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply the rotation with white background
        rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        
        return rotated
    
    def adjust_brightness(self, img, factor_range=(0.7, 1.3)):
        """
        Adjust the brightness of the impression area only.
        
        Args:
            img: Input image
            factor_range: Range of brightness adjustment factors
            
        Returns:
            Brightness-adjusted image
        """
        factor = random.uniform(*factor_range)
        
        # Create impression mask
        mask = self._create_impression_mask(img)
        mask_3ch = np.stack([mask, mask, mask], axis=2) if len(img.shape) == 3 else mask
        
        # Convert to HSV to modify brightness
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # Only apply to impression area
        result = img * (1 - mask_3ch) + adjusted * mask_3ch
        
        return result.astype(np.uint8)
    
    def adjust_contrast(self, img, factor_range=(0.7, 1.5)):
        """
        Adjust the contrast of the impression area only.
        
        Args:
            img: Input image
            factor_range: Range of contrast adjustment factors
            
        Returns:
            Contrast-adjusted image
        """
        factor = random.uniform(*factor_range)
        
        # Create impression mask
        mask = self._create_impression_mask(img)
        mask_3ch = np.stack([mask, mask, mask], axis=2) if len(img.shape) == 3 else mask
        
        # Calculate mean intensity
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        
        # Apply contrast adjustment
        adjusted = mean + factor * (img - mean)
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        
        # Only apply to impression area
        result = img * (1 - mask_3ch) + adjusted * mask_3ch
        
        return result.astype(np.uint8)
    
    def add_noise(self, img, noise_type="gaussian", amount_range=(0.01, 0.05)):
        """
        Add noise to the impression area only.
        
        Args:
            img: Input image
            noise_type: Type of noise ('gaussian', 'salt_pepper', 'speckle')
            amount_range: Range of noise intensity
            
        Returns:
            Noisy image
        """
        amount = random.uniform(*amount_range)
        
        # Create impression mask
        mask = self._create_impression_mask(img)
        mask_3ch = np.stack([mask, mask, mask], axis=2) if len(img.shape) == 3 else mask
        
        result = img.copy()
        
        if noise_type == "gaussian":
            # Gaussian noise
            mean = 0
            stddev = amount * 255
            noise = np.random.normal(mean, stddev, img.shape).astype(np.int16)
            noisy = cv2.add(img, noise, dtype=cv2.CV_8U)
            result = img * (1 - mask_3ch) + noisy * mask_3ch
            
        elif noise_type == "salt_pepper":
            # Salt and pepper noise
            salt_vs_pepper = 0.5  # Equal amount of salt and pepper noise
            
            # Create salt and pepper masks
            salt_mask = np.random.random(img.shape[:2]) < (amount * salt_vs_pepper)
            pepper_mask = np.random.random(img.shape[:2]) < (amount * (1 - salt_vs_pepper))
            
            # Apply only to impression area
            impression_pixels = (mask > 0)
            
            # Salt noise (white)
            salt_pixels = np.logical_and(salt_mask, impression_pixels)
            if len(img.shape) == 3:
                result[salt_pixels] = 255
            else:
                result[salt_pixels] = 255
            
            # Pepper noise (black)
            pepper_pixels = np.logical_and(pepper_mask, impression_pixels)
            if len(img.shape) == 3:
                result[pepper_pixels] = 0
            else:
                result[pepper_pixels] = 0
            
        elif noise_type == "speckle":
            # Speckle noise (multiplicative)
            noise = amount * np.random.randn(*img.shape)
            noisy = img + img * noise
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)
            result = img * (1 - mask_3ch) + noisy * mask_3ch
        
        return result.astype(np.uint8)
    
    def elastic_transform(self, img, alpha_range=(10, 50), sigma_range=(4, 6)):
        """
        Apply elastic deformation to the impression area only.
        
        Args:
            img: Input image
            alpha_range: Range of deformation intensity
            sigma_range: Range of deformation smoothness
            
        Returns:
            Elastically transformed image
        """
        try:
            alpha = random.uniform(*alpha_range)
            sigma = random.uniform(*sigma_range)
            
            # Save original shape
            shape = img.shape
            
            # Create impression mask
            mask = self._create_impression_mask(img)
            
            # Create random displacement fields
            dx = gaussian_filter((np.random.rand(*shape[:2]) * 2 - 1), sigma) * alpha
            dy = gaussian_filter((np.random.rand(*shape[:2]) * 2 - 1), sigma) * alpha
            
            # Create meshgrid for sampling
            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
            
            # Create distorted image (white background)
            distorted = np.ones_like(img) * 255
            
            # Apply deformation to each channel
            for i in range(shape[2] if len(shape) > 2 else 1):
                if len(shape) > 2:
                    channel = map_coordinates(img[:, :, i], indices, order=1, mode='constant', cval=255)
                    distorted[:, :, i] = channel.reshape(shape[:2])
                else:
                    channel = map_coordinates(img, indices, order=1, mode='constant', cval=255)
                    distorted = channel.reshape(shape)
            
            # Apply mask to only deform impression area
            mask_3ch = np.stack([mask, mask, mask], axis=2) if len(img.shape) == 3 else mask
            result = img * (1 - mask_3ch) + distorted * mask_3ch
            
            return result.astype(np.uint8)
            
        except Exception as e:
            print(f"Error in elastic transform: {str(e)}")
            return img
    
    def perspective_transform(self, img, strength_range=(0.03, 0.1)):
        """
        Apply perspective transformation while maintaining white background.
        
        Args:
            img: Input image
            strength_range: Range of transformation strength
            
        Returns:
            Perspective-transformed image
        """
        strength = random.uniform(*strength_range)
        h, w = img.shape[:2]
        
        # Calculate offset for each corner
        offset_h = int(h * strength)
        offset_w = int(w * strength)
        
        # Define source and destination points
        src_points = np.array([
            [0, 0],              # Top-left
            [w - 1, 0],          # Top-right
            [w - 1, h - 1],      # Bottom-right
            [0, h - 1]           # Bottom-left
        ], dtype=np.float32)
        
        dst_points = np.array([
            [random.randint(0, offset_w), random.randint(0, offset_h)],
            [w - 1 - random.randint(0, offset_w), random.randint(0, offset_h)],
            [w - 1 - random.randint(0, offset_w), h - 1 - random.randint(0, offset_h)],
            [random.randint(0, offset_w), h - 1 - random.randint(0, offset_h)]
        ], dtype=np.float32)
        
        # Compute the perspective transform matrix
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply the transform
        transformed = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        
        return transformed
    
    def cutout(self, img, num_holes_range=(3, 10), max_size_range=(10, 30)):
        """
        Apply cutout augmentation to simulate partial impressions.
        
        Args:
            img: Input image
            num_holes_range: Range for number of holes
            max_size_range: Range for maximum hole size
            
        Returns:
            Image with random cutouts
        """
        num_holes = random.randint(*num_holes_range)
        max_size = random.randint(*max_size_range)
        
        result = img.copy()
        
        # Create impression mask
        mask = self._create_impression_mask(img)
        
        # Get coordinates of impression pixels
        impression_coords = np.where(mask > 0.1)
        
        if len(impression_coords[0]) > 0:
            for _ in range(num_holes):
                # Randomly select a pixel within the impression
                idx = random.randint(0, len(impression_coords[0]) - 1)
                center_y, center_x = impression_coords[0][idx], impression_coords[1][idx]
                
                # Determine cutout size
                size_h = random.randint(5, max_size)
                size_w = random.randint(5, max_size)
                
                # Calculate cutout region boundaries
                y1 = max(0, center_y - size_h // 2)
                y2 = min(img.shape[0], center_y + size_h // 2)
                x1 = max(0, center_x - size_w // 2)
                x2 = min(img.shape[1], center_x + size_w // 2)
                
                # Fill the region with white
                result[y1:y2, x1:x2] = 255
        
        return result
    
    def simulate_smudge(self, img, strength_range=(0.3, 0.7)):
        """
        Simulate smudging effect on the impression.
        
        Args:
            img: Input image
            strength_range: Range of smudge effect strength
            
        Returns:
            Image with smudge effect
        """
        strength = random.uniform(*strength_range)
        
        # Create impression mask
        mask = self._create_impression_mask(img)
        mask_3ch = np.stack([mask, mask, mask], axis=2) if len(img.shape) == 3 else mask
        
        # Convert to PIL Image for filter application
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Apply motion blur
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=2))
        
        # Additional blur based on strength
        if strength > 0.5:
            pil_img = pil_img.filter(ImageFilter.BLUR)
        
        # Convert back to OpenCV format
        blurred = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # Apply smudge only to impression area
        alpha = strength  # Use strength directly as alpha
        blended = img * (1 - mask_3ch * alpha) + blurred * (mask_3ch * alpha)
        
        return blended.astype(np.uint8)
    
    def add_footprint_artifacts(self, img):
        """
        Add realistic footwear impression artifacts.
        
        Args:
            img: Input image
            
        Returns:
            Image with footprint artifacts
        """
        h, w = img.shape[:2]
        
        # Create impression mask
        mask = self._create_impression_mask(img)
        
        # Create artifact layer
        artifact = np.zeros((h, w), dtype=np.uint8)
        
        # Get coordinates of impression pixels
        impression_coords = np.where(mask > 0.1)
        
        if len(impression_coords[0]) > 0:
            # Create random curves for artifacts
            num_strokes = random.randint(3, 8)
            
            for _ in range(num_strokes):
                # Randomly select points within the impression
                indices = np.random.choice(len(impression_coords[0]), 4, replace=True)
                
                pt1 = (impression_coords[1][indices[0]], impression_coords[0][indices[0]])
                pt2 = (impression_coords[1][indices[1]], impression_coords[0][indices[1]])
                cp1 = (impression_coords[1][indices[2]], impression_coords[0][indices[2]])
                cp2 = (impression_coords[1][indices[3]], impression_coords[0][indices[3]])
                
                # Create Bezier curve points
                thickness = random.randint(1, 4)
                points = []
                
                for t in np.linspace(0, 1, 50):
                    # Bezier curve formula
                    point = ((1-t)**3 * np.array(pt1) + 
                            3*(1-t)**2*t * np.array(cp1) + 
                            3*(1-t)*t**2 * np.array(cp2) + 
                            t**3 * np.array(pt2))
                    points.append(tuple(point.astype(int)))
                
                # Draw curve
                for i in range(1, len(points)):
                    cv2.line(artifact, points[i-1], points[i], 255, thickness)
        
        # Blur the artifact to make it look more natural
        artifact = cv2.GaussianBlur(artifact, (5, 5), 0)
        
        # Convert to 3-channel
        artifact_3ch = np.stack([artifact] * 3, axis=2) if len(img.shape) == 3 else artifact
        
        # Create dilated mask for artifact application
        mask_dilated = cv2.dilate((mask * 255).astype(np.uint8), np.ones((5, 5), np.uint8))
        mask_dilated = mask_dilated / 255.0
        mask_dilated_3ch = np.stack([mask_dilated] * 3, axis=2) if len(img.shape) == 3 else mask_dilated
        
        # Calculate artifact influence
        alpha = random.uniform(0.1, 0.3)
        artifact_influence = (artifact_3ch / 255.0) * alpha * mask_dilated_3ch
        
        # Apply artifacts with dark gray to black color
        color = random.randint(20, 120)
        color_artifact = np.ones_like(img) * color if len(img.shape) == 3 else color
        
        # Blend with original image
        blended = img * (1 - artifact_influence) + color_artifact * artifact_influence
        
        return blended.astype(np.uint8)
    
    def random_lines(self, img, num_lines_range=(3, 8)):
        """
        Add random lines to simulate scratches or marks.
        
        Args:
            img: Input image
            num_lines_range: Range for number of lines
            
        Returns:
            Image with random lines
        """
        num_lines = random.randint(*num_lines_range)
        result = img.copy()
        
        # Create impression mask
        mask = self._create_impression_mask(img)
        
        # Get coordinates of impression pixels
        impression_coords = np.where(mask > 0.1)
        
        if len(impression_coords[0]) > 0:
            for _ in range(num_lines):
                # Randomly select start and end points within the impression
                indices = np.random.choice(len(impression_coords[0]), 2, replace=True)
                
                start_y, start_x = impression_coords[0][indices[0]], impression_coords[1][indices[0]]
                end_y, end_x = impression_coords[0][indices[1]], impression_coords[1][indices[1]]
                
                # Line thickness
                thickness = random.randint(1, 3)
                
                # Line color (dark)
                color = random.randint(30, 150)
                color_tuple = (color, color, color) if len(img.shape) == 3 else color
                
                # Draw line
                cv2.line(result, (start_x, start_y), (end_x, end_y), color_tuple, thickness)
        
        return result
    
    def augment(self, img):
        """
        Apply random augmentations to the image.
        
        Args:
            img: Input image
            
        Returns:
            Augmented image
        """
        # Make a copy of the input image
        result = img.copy()
        
        # Apply random augmentations based on probability
        augmentations = []
        
        # Rotation (common)
        if random.random() < self.aug_probs['rotate']:
            result = self.rotate_image(result)
            augmentations.append('rotate')
        
        # Brightness adjustment
        if random.random() < self.aug_probs['brightness']:
            result = self.adjust_brightness(result)
            augmentations.append('brightness')
        
        # Contrast adjustment
        if random.random() < self.aug_probs['contrast']:
            result = self.adjust_contrast(result)
            augmentations.append('contrast')
        
        # Noise addition
        if random.random() < self.aug_probs['noise']:
            noise_type = random.choice(['gaussian', 'salt_pepper', 'speckle'])
            result = self.add_noise(result, noise_type)
            augmentations.append(f'noise_{noise_type}')
        
        # Elastic transformation (less common)
        if random.random() < self.aug_probs['elastic']:
            result = self.elastic_transform(result)
            augmentations.append('elastic')
        
        # Perspective transformation
        if random.random() < self.aug_probs['perspective']:
            result = self.perspective_transform(result)
            augmentations.append('perspective')
        
        # Cutout simulation
        if random.random() < self.aug_probs['cutout']:
            result = self.cutout(result)
            augmentations.append('cutout')
        
        # Smudge effect
        if random.random() < self.aug_probs['smudge']:
            result = self.simulate_smudge(result)
            augmentations.append('smudge')
        
        # Footprint artifacts
        if random.random() < self.aug_probs['artifacts']:
            result = self.add_footprint_artifacts(result)
            augmentations.append('artifacts')
        
        # Random lines
        if random.random() < self.aug_probs['random_lines']:
            result = self.random_lines(result)
            augmentations.append('random_lines')
        
        return result, augmentations
    
    def create_augmentation_batch(self, img, num_variants=5):
        """
        Create a batch of augmented variants of an image.
        
        Args:
            img: Input image
            num_variants: Number of augmented variants to create
            
        Returns:
            List of augmented images and their augmentation descriptions
        """
        augmented_batch = []
        
        # Include original image
        augmented_batch.append((img.copy(), ['original']))
        
        # Create augmented variants
        for _ in range(num_variants):
            aug_img, aug_names = self.augment(img)
            augmented_batch.append((aug_img, aug_names))
        
        return augmented_batch


# Demo/testing code
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sys
    
    # Check if input image is provided
    if len(sys.argv) < 2:
        print("Usage: python augmentation.py <input_image_path>")
        sys.exit(1)
    
    # Load test image
    input_path = sys.argv[1]
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"Error: Could not load image {input_path}")
        sys.exit(1)
    
    # Create augmenter and generate augmented batch
    augmenter = FootwearAugmenter(seed=42)
    augmented_batch = augmenter.create_augmentation_batch(img, num_variants=5)
    
    # Visualize results
    plt.figure(figsize=(20, 10))
    
    for i, (aug_img, aug_names) in enumerate(augmented_batch):
        plt.subplot(2, 3, i+1)
        plt.imshow(cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB))
        plt.title(', '.join(aug_names))
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_examples.png')
    plt.show()
