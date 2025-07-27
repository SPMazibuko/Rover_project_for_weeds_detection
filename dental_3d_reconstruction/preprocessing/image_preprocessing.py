"""
Dental X-ray Image Preprocessing Module
Implements CLAHE enhancement and ROI extraction for panoramic dental X-rays
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import torch
from skimage import exposure, filters
from scipy import ndimage


class DentalImagePreprocessor:
    """
    Advanced preprocessing pipeline for panoramic dental X-ray images.
    Includes CLAHE enhancement, noise reduction, and ROI extraction.
    """
    
    def __init__(self, 
                 clahe_clip_limit: float = 2.0,
                 clahe_tile_grid_size: Tuple[int, int] = (8, 8),
                 target_size: Tuple[int, int] = (512, 512),
                 gaussian_sigma: float = 1.0):
        """
        Initialize the preprocessor with configurable parameters.
        
        Args:
            clahe_clip_limit: Clipping limit for CLAHE
            clahe_tile_grid_size: Grid size for CLAHE tiles
            target_size: Target output image size
            gaussian_sigma: Sigma for Gaussian blur denoising
        """
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.target_size = target_size
        self.gaussian_sigma = gaussian_sigma
        
        # Initialize CLAHE processor
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid_size
        )
    
    def apply_clahe_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
        
        Args:
            image: Input grayscale image
            
        Returns:
            Enhanced image with improved local contrast
        """
        if len(image.shape) == 3:
            # Convert to grayscale if needed
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Apply CLAHE
        enhanced = self.clahe.apply(image)
        return enhanced
    
    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply denoising using Gaussian filter and median filter.
        
        Args:
            image: Input image
            
        Returns:
            Denoised image
        """
        # Gaussian blur for noise reduction
        denoised = filters.gaussian(image, sigma=self.gaussian_sigma)
        
        # Median filter for salt-and-pepper noise
        denoised = ndimage.median_filter(denoised, size=3)
        
        return denoised
    
    def extract_dental_roi(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Extract Region of Interest (ROI) containing dental structures.
        
        Args:
            image: Preprocessed dental X-ray image
            
        Returns:
            Tuple of (ROI image, bounding box coordinates)
        """
        # Apply morphological operations to find dental arch
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        
        # Threshold to create binary mask
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological closing to fill gaps
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find largest contour (dental arch)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get bounding box of largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add padding
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            
            roi = image[y:y+h, x:x+w]
            bbox = (x, y, w, h)
        else:
            # If no contour found, use center region
            h, w = image.shape[:2]
            x, y = w // 4, h // 4
            roi_w, roi_h = w // 2, h // 2
            roi = image[y:y+roi_h, x:x+roi_w]
            bbox = (x, y, roi_w, roi_h)
        
        return roi, bbox
    
    def resize_and_normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size and normalize values.
        
        Args:
            image: Input image
            
        Returns:
            Resized and normalized image
        """
        # Resize to target size
        resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_CUBIC)
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
    
    def preprocess_image(self, image: np.ndarray, extract_roi: bool = True) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
        """
        Complete preprocessing pipeline for dental X-ray images.
        
        Args:
            image: Input raw dental X-ray image
            extract_roi: Whether to extract ROI
            
        Returns:
            Tuple of (preprocessed image, ROI bounding box if extracted)
        """
        # Step 1: Apply CLAHE enhancement
        enhanced = self.apply_clahe_enhancement(image)
        
        # Step 2: Denoise
        denoised = self.denoise_image(enhanced)
        
        # Step 3: Extract ROI if requested
        bbox = None
        if extract_roi:
            processed_image, bbox = self.extract_dental_roi(denoised)
        else:
            processed_image = denoised
        
        # Step 4: Resize and normalize
        final_image = self.resize_and_normalize(processed_image)
        
        return final_image, bbox
    
    def preprocess_batch(self, images: np.ndarray) -> Tuple[torch.Tensor, list]:
        """
        Preprocess a batch of images.
        
        Args:
            images: Batch of input images [N, H, W] or [N, H, W, C]
            
        Returns:
            Tuple of (preprocessed tensor, list of bounding boxes)
        """
        processed_images = []
        bboxes = []
        
        for image in images:
            processed, bbox = self.preprocess_image(image)
            processed_images.append(processed)
            bboxes.append(bbox)
        
        # Convert to tensor and add channel dimension
        tensor = torch.FloatTensor(np.stack(processed_images))
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(1)  # Add channel dimension
        
        return tensor, bboxes
