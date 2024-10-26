import torch
import paddle
import logging
import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from paddleocr import PaddleOCR

# Import necessary classes from training.py
from training import OCRSetup, ImageProcessor, GeometryCalculator
from training import MeasurementExtractor, DimensionProcessor, ProductAnalyzer

def process_image_extraction(image, entity_name):
    """
    Process a single image to extract the specified entity information.
    
    Args:
        image_path (str): URL or local path to the image
        entity_name (str): Type of entity to extract (e.g., 'height', 'weight', 'voltage', etc.)
        
    Returns:
        tuple: (extracted_value, ocr_text)
            - extracted_value: The extracted measurement with units
            - ocr_text: Raw OCR text from the image
    """
    try:
        # Initialize the product analyzer
        if validate_input(image, entity_name):
            analyzer = ProductAnalyzer()
            
            # Process the image
            image_flag = True
            result, ocr_text = analyzer.process_single_image(image, entity_name, image_flag=image_flag)
            
            return {
                "status": "success",
                "entity_name": entity_name,
                "extracted_value": result,
                "ocr_text": ocr_text
            }
        else:
            return {
                'status': "error",
                'message': "Invalid Input"
            }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

def validate_input(image, entity_name):
    """
    Validate the input parameters.
    
    Args:
        image_path (str): URL or local path to the image
        entity_name (str): Type of entity to extract
        
    Returns:
        bool: True if input is valid, False otherwise
    """
    valid_entities = [
        'depth', 'height', 'width', 'item_weight',
        'wattage', 'voltage', 'item_volume',
        'maximum_weight_recommendation'
    ]
    
    if not image:
        print("Error: Image is required")
        return False
        
    if entity_name not in valid_entities:
        print(f"Error: Invalid entity name. Must be one of: {', '.join(valid_entities)}")
        return False
        
    return True

# if __name__ == "__main__":
#     import argparse
    
#     # Set up argument parser
#     parser = argparse.ArgumentParser(description='Extract product measurements from images')
#     parser.add_argument('--image', required=True, help='URL or path to the image')
#     parser.add_argument('--entity', required=True, help='Entity type to extract (e.g., height, weight, voltage)')
    
#     # Parse arguments
#     args = parser.parse_args()
    
#     # Validate input
#     if validate_input(args.image, args.entity):
#         # Process the image
#         result = process_image(args.image, args.entity)
        
#         # Print results
#         if result["status"] == "success":
#             print("\nResults:")
#             print(f"Entity Name: {result['entity_name']}")
#             print(f"Extracted Value: {result['extracted_value']}")
#             print(f"OCR Text: {result['ocr_text']}")
#         else:
#             print(f"\nError: {result['message']}")

# Example usage:
# python inference.py --image "https://example.com/product.jpg" --entity "height"
# python inference.py --image "local_image.jpg" --entity "weight"