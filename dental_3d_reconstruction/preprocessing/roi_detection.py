"""
ROI Detection Module for Dental X-ray Images
Advanced region of interest detection using anatomical landmarks
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
from sklearn.cluster import DBSCAN
import torch
import torch.nn as nn


class ROIDetector:
    """
    Advanced ROI detector for dental panoramic X-rays.
    Uses template matching and anatomical landmark detection.
    """
    
    def __init__(self, template_threshold: float = 0.7):
        """
        Initialize ROI detector.
        
        Args:
            template_threshold: Threshold for template matching
        """
        self.template_threshold = template_threshold
        self._create_dental_templates()
    
    def _create_dental_templates(self):
        """Create template patterns for different tooth types."""
        # Create basic tooth templates (simplified patterns)
        self.molar_template = self._create_molar_template()
        self.incisor_template = self._create_incisor_template()
        self.canine_template = self._create_canine_template()
    
    def _create_molar_template(self) -> np.ndarray:
        """Create a molar tooth template."""
        template = np.zeros((40, 30), dtype=np.uint8)
        # Create molar-like shape
        cv2.rectangle(template, (5, 10), (25, 35), 255, -1)
        cv2.rectangle(template, (8, 5), (22, 10), 255, -1)
        # Add cusps
        cv2.circle(template, (12, 8), 3, 255, -1)
        cv2.circle(template, (18, 8), 3, 255, -1)
        return template
    
    def _create_incisor_template(self) -> np.ndarray:
        """Create an incisor tooth template."""
        template = np.zeros((35, 20), dtype=np.uint8)
        # Create incisor-like shape
        cv2.rectangle(template, (5, 10), (15, 30), 255, -1)
        # Triangular top
        pts = np.array([[5, 10], [15, 10], [10, 5]], np.int32)
        cv2.fillPoly(template, [pts], 255)
        return template
    
    def _create_canine_template(self) -> np.ndarray:
        """Create a canine tooth template."""
        template = np.zeros((40, 25), dtype=np.uint8)
        # Create canine-like shape
        cv2.rectangle(template, (5, 15), (20, 35), 255, -1)
        # Pointed top
        pts = np.array([[5, 15], [20, 15], [12, 5]], np.int32)
        cv2.fillPoly(template, [pts], 255)
        return template
    
    def detect_teeth_landmarks(self, image: np.ndarray) -> List[Dict]:
        """
        Detect individual teeth landmarks using template matching.
        
        Args:
            image: Preprocessed dental X-ray image
            
        Returns:
            List of detected teeth with their properties
        """
        teeth_detections = []
        
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        templates = [
            (self.molar_template, 'molar'),
            (self.incisor_template, 'incisor'),
            (self.canine_template, 'canine')
        ]
        
        for template, tooth_type in templates:
            # Multi-scale template matching
            scales = [0.8, 1.0, 1.2, 1.4]
            
            for scale in scales:
                # Resize template
                scaled_template = cv2.resize(
                    template, 
                    None, 
                    fx=scale, 
                    fy=scale, 
                    interpolation=cv2.INTER_CUBIC
                )
                
                if scaled_template.shape[0] > image.shape[0] or scaled_template.shape[1] > image.shape[1]:
                    continue
                
                # Template matching
                result = cv2.matchTemplate(image, scaled_template, cv2.TM_CCOEFF_NORMED)
                locations = np.where(result >= self.template_threshold)
                
                for pt in zip(*locations[::-1]):
                    teeth_detections.append({
                        'position': pt,
                        'type': tooth_type,
                        'scale': scale,
                        'confidence': result[pt[1], pt[0]],
                        'size': scaled_template.shape
                    })
        
        # Remove overlapping detections
        teeth_detections = self._remove_overlapping_detections(teeth_detections)
        
        return teeth_detections
    
    def _remove_overlapping_detections(self, detections: List[Dict]) -> List[Dict]:
        """Remove overlapping tooth detections using Non-Maximum Suppression."""
        if not detections:
            return []
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        filtered_detections = []
        for detection in detections:
            is_overlapping = False
            x1, y1 = detection['position']
            h1, w1 = detection['size']
            
            for existing in filtered_detections:
                x2, y2 = existing['position']
                h2, w2 = existing['size']
                
                # Calculate overlap
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y
                
                area1 = w1 * h1
                area2 = w2 * h2
                union_area = area1 + area2 - overlap_area
                
                if union_area > 0 and overlap_area / union_area > 0.3:
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                filtered_detections.append(detection)
        
        return filtered_detections
    
    def detect_dental_arch(self, image: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Detect the dental arch curve and key points.
        
        Args:
            image: Preprocessed dental X-ray image
            
        Returns:
            Tuple of (arch mask, arch key points)
        """
        # Edge detection
        edges = cv2.Canny(image, 50, 150)
        
        # Morphological operations to connect arch
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by arch-like properties
        arch_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum area threshold
                # Check if contour resembles an arch
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                
                if 0.3 < solidity < 0.8:  # Arch-like solidity
                    arch_contours.append(contour)
        
        # Create arch mask
        arch_mask = np.zeros_like(image)
        if arch_contours:
            # Use the largest qualifying contour
            main_arch = max(arch_contours, key=cv2.contourArea)
            cv2.drawContours(arch_mask, [main_arch], -1, 255, thickness=cv2.FILLED)
            
            # Extract key points along the arch
            arch_points = self._extract_arch_keypoints(main_arch)
        else:
            arch_points = []
        
        return arch_mask, arch_points
    
    def _extract_arch_keypoints(self, contour: np.ndarray, num_points: int = 20) -> List[Tuple[int, int]]:
        """Extract evenly spaced keypoints along the dental arch."""
        # Approximate contour to reduce points
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Extract points
        points = [tuple(point[0]) for point in approx]
        
        # If we have too many points, subsample
        if len(points) > num_points:
            indices = np.linspace(0, len(points) - 1, num_points, dtype=int)
            points = [points[i] for i in indices]
        
        return points
    
    def create_roi_mask(self, image: np.ndarray, teeth_detections: List[Dict]) -> np.ndarray:
        """
        Create ROI mask based on detected teeth landmarks.
        
        Args:
            image: Input dental X-ray image
            teeth_detections: List of detected teeth
            
        Returns:
            Binary ROI mask
        """
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        if not teeth_detections:
            # If no teeth detected, use center region
            h, w = image.shape[:2]
            cv2.rectangle(mask, (w//4, h//4), (3*w//4, 3*h//4), 255, -1)
            return mask
        
        # Create bounding box around all detected teeth
        all_points = []
        for detection in teeth_detections:
            x, y = detection['position']
            h, w = detection['size']
            all_points.extend([(x, y), (x + w, y + h)])
        
        if all_points:
            # Find bounding box of all teeth
            xs, ys = zip(*all_points)
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            # Add padding
            padding = 30
            min_x = max(0, min_x - padding)
            min_y = max(0, min_y - padding)
            max_x = min(image.shape[1], max_x + padding)
            max_y = min(image.shape[0], max_y + padding)
            
            # Create rectangular ROI
            cv2.rectangle(mask, (min_x, min_y), (max_x, max_y), 255, -1)
        
        return mask
