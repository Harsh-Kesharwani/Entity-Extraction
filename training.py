# Import required libraries
import torch
import paddle
import logging
import cv2
import time
import math
import numpy as np
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from pathlib import Path
from paddleocr import PaddleOCR
import re
from tqdm import tqdm
import matplotlib.pyplot as plt

class OCRSetup:
    """Initialize OCR and device settings"""
    @staticmethod
    def setup_environment():
        # Set device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initialize PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)

        # Configure logging
        logging.getLogger('ppocr').setLevel(logging.WARNING)

        # Print device info
        print("OCR GPU Compile Check: ", paddle.device.is_compiled_with_cuda())
        print("OCR on GPU check: ", paddle.device.get_device())
        print("Current device: ", device)

        return device, ocr

class ImageProcessor:
    """Handle image processing operations"""
    @staticmethod
    def load_image_from_url(url):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                img_BGR = np.array(img)
                return img_BGR
            return None
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    @staticmethod
    def calculate_centroid(box):
        sum_x, sum_y = 0, 0
        for point in box:
            sum_x += point[0]
            sum_y += point[1]
        return (sum_x // len(box), sum_y // len(box))

class GeometryCalculator:
    """Handle geometric calculations"""
    @staticmethod
    def distance_point_to_line(px, py, x1, y1, x2, y2):
        numerator = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
        denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        return numerator / denominator

    @staticmethod
    def line_angle(x1, y1, x2, y2):
        return abs(math.degrees(math.atan2(y2 - y1, x2 - x1))) % 360

    @staticmethod
    def is_approximately_vertical(angle, lower_bound=75, upper_bound=105):
        return lower_bound <= angle <= upper_bound

    @staticmethod
    def is_approximately_horizontal(angle, lower_bound=0, upper_bound=60):
        return lower_bound <= angle <= upper_bound

class MeasurementExtractor:
    """Extract various measurements from text"""
    @staticmethod
    def format_number(num):
        return '{:.2f}'.format(num).rstrip('0').rstrip('.')

    @staticmethod
    def extract_number(text):
        match = re.search(r'\d+(\.\d+)?', text)
        return float(match.group()) if match else None

    @staticmethod
    def extract_weight(text):
        text = str(text)
        text = text.replace(',', '.')
        numbers = re.findall(r'(\d+\.?\d*)\s*([a-zA-Z]+)', text, re.IGNORECASE)
        
        if not numbers:
            return " "
        
        measurements = {
            'mg': ('milligram', None),
            'g': ('gram', None),
            'kg': ('kilogram', None),
            'oz': ('ounce', None),
            'lb': ('pound', None),
            'ton': ('ton', None)
        }
        
        for number, unit in numbers:
            num = float(number)
            unit = unit.lower()
            
            for key, (name, value) in measurements.items():
                if unit.startswith(key):
                    if value is None or num > value:
                        measurements[key] = (name, num)
        
        # Return the largest measurement found
        for key, (name, value) in measurements.items():
            if value is not None:
                return f"{MeasurementExtractor.format_number(value)} {name}"
        
        return " "

    @staticmethod
    def extract_wattage(text):
        text = str(text)
        numbers = re.findall(r'(\d+\.?\d*)\s*([a-zA-Z]+)', text, re.IGNORECASE)

        if not numbers:
            return " "

        largest_watt = None
        largest_kilowatt = None

        for number, unit in numbers:
            num = float(number)
            unit = unit.lower()

            if unit.startswith('kw') or unit.startswith('kwh') or unit.startswith('kilowatt'):
                largest_kilowatt = max(num, largest_kilowatt) if largest_kilowatt else num
            elif unit.startswith('w') or unit.startswith('watt'):
                largest_watt = max(num, largest_watt) if largest_watt else num

        if largest_watt is not None:
            return f"{largest_watt} watt"
        elif largest_kilowatt is not None:
            return f"{largest_kilowatt} kilowatt"
        return " "

    @staticmethod
    def extract_voltage(text):
        # Implementation as in your original code
        text = str(text)
        numbers = re.findall(r'(\d+\.?\d*)\s*([a-zA-Z]+)', text, re.IGNORECASE)

        if not numbers:
            return " "

        measurements = {
            'mv': ('millivolt', None),
            'v': ('volt', None),
            'kv': ('kilovolt', None)
        }

        for number, unit in numbers:
            num = float(number)
            unit = unit.lower()

            for key, (name, value) in measurements.items():
                if unit.startswith(key):
                    if value is None or num > value:
                        measurements[key] = (name, num)

        # Return the largest measurement found
        for key, (name, value) in measurements.items():
            if value is not None:
                return f"{MeasurementExtractor.format_number(value)} {name}"

        return " "

    @staticmethod
    def extract_volume(text):
        # Implementation as in your original code
        text = str(text)
        numbers = re.findall(r'(\d+\.?\d*)\s*([a-zA-Z0"\']+)', text, re.IGNORECASE)

        if not numbers:
            return " "

        measurements = {
            'ml': ('millilitre', None),
            'l': ('litre', None),
            'oz': ('fluid ounce', None),
            'fl': ('fluid ounce', None),
            'gal': ('gallon', None),
            'pt': ('pint', None),
            'qt': ('quart', None),
            'dl': ('decilitre', None),
            'cup': ('cup', None),
            'ft': ('cubic foot', None),
            'in': ('cubic inch', None)
        }

        for number, unit in numbers:
            num = float(number)
            unit = unit.lower()

            for key, (name, value) in measurements.items():
                if unit.startswith(key):
                    if value is None or num > value:
                        measurements[key] = (name, num)

        # Return the largest measurement found
        for key, (name, value) in measurements.items():
            if value is not None:
                return f"{MeasurementExtractor.format_number(value)} {name}"

        return " "

class DimensionProcessor:
    """Process dimensions from OCR results"""
    def __init__(self, ocr):
        self.ocr = ocr
        self.units_map = {
            'mm': 'millimetre',
            'millimetre': 'millimetre',
            'cm': 'centimetre',
            'centimetre': 'centimetre',
            'in': 'inch',
            '"': 'inch',
            'inch': 'inch',
            "'": 'foot',
            'ft': 'foot',
            'foot': 'foot',
            'yd': 'yard',
            'yard': 'yard',
            'metre': 'metre',
        }

    def compute_dimension(self, image_url, mode="h", image_flag=False):
        if image_flag:
            img_BGR=np.array(image_url) #img_url is image
        else:
            img_BGR = ImageProcessor.load_image_from_url(image_url)
        if img_BGR is None:
            return ("-1", "")

        # Get image center
        height, width, _ = img_BGR.shape
        center_x, center_y = width // 2, height // 2

        # Perform OCR
        result = self.ocr.ocr(img_BGR, cls=True)
        if not result or result[0] is None:
            return ("-1", "")

        # Process OCR results
        dims, unit_names, dim_scores, dim_bboxes = [], [], [], []
        texts = [line[1][0] for line in result[0]]

        for line in result:
            for word in line:
                bbox, dim, score = word[0], word[1][0], word[1][1]
                for unit, full_name in self.units_map.items():
                    if unit in dim.lower():
                        unit_names.append(full_name)
                        dim_scores.append(score)
                        dim_bboxes.append(bbox)
                        dims.append(dim)

        # Find relevant dimensions based on mode
        candidates = self._process_lines_and_angles(img_BGR, dim_bboxes, dims, unit_names, mode)

        # Get final dimension
        max_dim = -1
        max_dim_name = ""
        for _, dim, name in candidates:
            cur_dim = MeasurementExtractor.extract_number(dim)
            if cur_dim is not None and cur_dim > max_dim:
                max_dim = cur_dim
                max_dim_name = name

        return (f"{max_dim} {max_dim_name}", texts)

    def _process_lines_and_angles(self, image, bboxes, dims, unit_names, mode):
        # Convert to grayscale and detect edges
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

        if lines is None:
            return []

        # Calculate centers of bounding boxes
        bbox_centers = [ImageProcessor.calculate_centroid(box) for box in bboxes]
        candidates = []

        for i, bbox_center in enumerate(bbox_centers):
            min_distance = float('inf')
            closest_line = None

            # Find closest line to each bbox center
            for line in lines:
                x1, y1, x2, y2 = line[0]
                px, py = bbox_center
                dist = GeometryCalculator.distance_point_to_line(px, py, x1, y1, x2, y2)

                if dist < min_distance:
                    min_distance = dist
                    closest_line = line[0]

            if closest_line is not None:
                x1, y1, x2, y2 = closest_line
                angle = GeometryCalculator.line_angle(x1, y1, x2, y2)

                # Check line orientation based on mode
                if ((mode == "h" and GeometryCalculator.is_approximately_vertical(angle)) or
                    (mode in ["l", "w"] and GeometryCalculator.is_approximately_horizontal(angle))):
                    candidates.append((bbox_center, dims[i], unit_names[i]))

        return candidates

class ProductAnalyzer:
    """Main class to analyze product images"""
    def __init__(self):
        self.device, self.ocr = OCRSetup.setup_environment()
        self.dimension_processor = DimensionProcessor(self.ocr)
        self.vision_model_calls=0

    def process_single_image(self, image_url, entity_name, image_flag=False):
        """Process a single image for a given entity"""
        # Determine processing pipeline based on entity name
        if image_flag: 
            if entity_name in ['depth', 'height', 'width']:
                mode = 'h' if entity_name in ['depth', 'height'] else 'w'
                return self.dimension_processor.compute_dimension(image_url, mode=mode, image_flag=True)
        else:
            if entity_name in ['depth', 'height', 'width']:
                mode = 'h' if entity_name in ['depth', 'height'] else 'w'
                return self.dimension_processor.compute_dimension(image_url, mode=mode)

        # Get OCR text for measurement extraction
        if image_flag:
            img_BGR=np.array(image_url)
        else:
            img_BGR = ImageProcessor.load_image_from_url(image_url)
        
        if img_BGR is None:
            return (" ", "")

        result = self.ocr.ocr(img_BGR, cls=True)
        if not result or not result[0]:
            return (" ", "")

        ocr_text = ' '.join([line[1][0] for line in result[0]])

        # Extract measurements based on entity type
        if entity_name == 'item_weight':
            return (MeasurementExtractor.extract_weight(ocr_text), ocr_text)

        elif entity_name == 'wattage':
            return (MeasurementExtractor.extract_wattage(ocr_text), ocr_text)

        elif entity_name == 'voltage':
            return (MeasurementExtractor.extract_voltage(ocr_text), ocr_text)

        elif entity_name == 'item_volume':
            return (MeasurementExtractor.extract_volume(ocr_text), ocr_text)

        elif entity_name == 'maximum_weight_recommendation':
            return (MeasurementExtractor.extract_weight(ocr_text), ocr_text)

        return (" ", "")

    def process_dataset(self, input_file, batch_size=500):
        """Process entire dataset"""
        # Read input file
        df = pd.read_csv(input_file)
        df=df[0:100]
        df.rename(columns={'entity_value': 'true_entity_value'}, inplace=True)

        # Create copy for OCR output
        df_ocr = df.copy()

        # Initialize counters
        count_correct = 0
        count_wrong = 0

        # Process each row
        for index, row in tqdm(df.iterrows(), total=len(df)):
            try:
                value, ocr_text = self.process_single_image(row['image_link'], row['entity_name'])

                # Store results
                df.at[index, 'entity_value'] = value
                df_ocr.at[index, 'OCR Output'] = ocr_text

                # Update counters
                if value and value != " " and "-1" not in value:
                    count_correct += 1
                else:
                    count_wrong += 1

                # Save intermediate results
                if (index + 1) % batch_size == 0 or (index + 1) == len(df):
                    df.to_csv(f'./dataset/output/test_results.csv', index=False)
                    df_ocr.to_csv(f'./dataset/output/test_OCR.csv', index=False)
                    print(f"Progress: {index+1}/{len(df)}")
                    print(f"Correct: {count_correct}, Wrong: {count_wrong}")

            except Exception as e:
                print(f"Error processing row {index}: {e}")

        # Save final results
        df.to_csv(f'./dataset/output/test_results.csv', index=False)
        df_ocr.to_csv(f'./dataset/output/test_OCR.csv', index=False)

        print(f"Final Progress: {len(df)}/{len(df)}")
        print(f"Final Correct: {count_correct}, Final Wrong: {count_wrong}")

product_analyzer = ProductAnalyzer()
product_analyzer.process_dataset('./dataset/input/train.csv')
