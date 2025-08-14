"""
Patent-Based Material Recognition Model
Based on: 一种基于AI图形识别技术的数字面料物理属性获取方法

Key Features:
1. Dual-view image processing (top-view and side-view)
2. Multi-modal architecture combining visual and physical parameters
3. Parameter importance weighting based on patent findings
4. Physics-informed loss function
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Concatenate, Dropout, 
                                   GlobalAveragePooling2D, Multiply, BatchNormalization)
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import cv2
import imghdr
from collections import Counter

class PatentDataLoader:
    """
    Data loader following patent specifications:
    - 30cm×30cm fabric samples on 18cm diameter, 40cm height platform
    - Dual-view capture (top-view: -1, side-view: -2)
    - Physical parameters: [弯曲强度, 强度, 形变强度, 形变率]
    """
    
    def __init__(self, data_path, image_size=(224, 224)):
        self.data_path = data_path
        self.image_size = image_size
        
        # Parameter importance weights based on patent findings
        # 弯曲强度(最显著), 强度(较显著), 变形强度(较显著), 变形率(不显著)
        self.parameter_weights = np.array([1.0, 0.6, 0.6, 0.3], dtype=np.float32)
        
    def extract_parameters_from_folder(self, folder_name):
        """
        Extract physical parameters from folder name
        Format: "1-20.20.20.20-1" -> [弯曲强度, 强度, 形变强度, 形变率]
        """
        try:
            parts = folder_name.split('-')
            if len(parts) >= 2:
                param_str = parts[1]  # "20.20.20.20"
                params = [float(x) for x in param_str.split('.')]
                if len(params) >= 4:
                    # Validate parameter ranges (based on patent typical values)
                    validated_params = []
                    for i, param in enumerate(params[:4]):
                        if i == 0:  # 弯曲强度 (0-100)
                            validated_params.append(max(0, min(100, param)))
                        else:  # Other parameters (0-100) 
                            validated_params.append(max(0, min(100, param)))
                    return validated_params
        except Exception as e:
            print(f"Error extracting parameters from {folder_name}: {e}")
        
        return [0.0, 20.0, 20.0, 20.0]  # Default parameters
    
    def load_and_preprocess_image(self, image_path):
        """Load and preprocess image according to patent specs"""
        try:
            if not os.path.exists(image_path):
                return None
                
            # Check if valid image
            if imghdr.what(image_path) not in ['jpeg', 'jpg', 'png']:
                return None
                
            # Load image as RGB (patent specifies color capture)
            img = load_img(image_path, target_size=self.image_size, color_mode='rgb')
            img_array = img_to_array(img) / 255.0  # Normalize to [0,1]
            return img_array
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def create_parameter_based_labels(self, parameters):
        """
        Create labels based on parameter similarity following patent methodology
        Groups materials with similar physical properties using broader ranges
        """
        param_labels = []
        
        for params in parameters:
            # Create broader groupings for better class balance
            # Based on patent findings about parameter importance
            
            # Group 弯曲强度 (most significant) into broader ranges
            if params[0] <= 25:
                bending_group = "Low"      # 0-25
            elif params[0] <= 50: 
                bending_group = "Medium"   # 26-50
            elif params[0] <= 75:
                bending_group = "High"     # 51-75
            else:
                bending_group = "VeryHigh" # 76-100
            
            # Group 强度 (moderately significant) 
            if params[1] <= 40:
                strength_group = "Low"     # 0-40
            elif params[1] <= 70:
                strength_group = "Medium"  # 41-70  
            else:
                strength_group = "High"    # 71-100
            
            # Group 形变强度 (moderately significant)
            if params[2] <= 30:
                deform_strength_group = "Low"    # 0-30
            elif params[2] <= 50:
                deform_strength_group = "Medium" # 31-50
            else:
                deform_strength_group = "High"   # 51-100
            
            # Group 形变率 (not significant) - broader groups
            if params[3] <= 50:
                deform_rate_group = "Low"   # 0-50
            else:
                deform_rate_group = "High"  # 51-100
            
            # Create composite label focusing on most important parameters
            param_label = f"Bend_{bending_group}_Str_{strength_group}_DefStr_{deform_strength_group}_DefRate_{deform_rate_group}"
            param_labels.append(param_label)
        
        return param_labels

    def load_dual_view_data(self):
        """
        Load dual-view images with corresponding physical parameters
        Returns: top_images, side_images, parameters, labels
        """
        top_images = []
        side_images = []
        parameters = []
        original_labels = []
        label_names = []
        
        print("Loading patent-based dual-view data...")
        
        for folder in sorted(os.listdir(self.data_path)):
            folder_path = os.path.join(self.data_path, folder)
            
            if not os.path.isdir(folder_path):
                continue
                
            # Extract physical parameters from folder name
            params = self.extract_parameters_from_folder(folder)
            
            # Find image pairs (top-view and side-view)
            files = os.listdir(folder_path)
            image_pairs = {}
            
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    if '-1.' in file:  # Top view
                        base_name = file.replace('-1.', '.')
                        if base_name not in image_pairs:
                            image_pairs[base_name] = {}
                        image_pairs[base_name]['top'] = file
                    elif '-2.' in file:  # Side view
                        base_name = file.replace('-2.', '.')
                        if base_name not in image_pairs:
                            image_pairs[base_name] = {}
                        image_pairs[base_name]['side'] = file
            
            # Load valid pairs
            for base_name, pair in image_pairs.items():
                if 'top' in pair and 'side' in pair:
                    top_path = os.path.join(folder_path, pair['top'])
                    side_path = os.path.join(folder_path, pair['side'])
                    
                    top_img = self.load_and_preprocess_image(top_path)
                    side_img = self.load_and_preprocess_image(side_path)
                    
                    if top_img is not None and side_img is not None:
                        top_images.append(top_img)
                        side_images.append(side_img)
                        parameters.append(params)
                        original_labels.append(folder)
                        label_names.append(f"{folder}_{base_name}")
        
        # Create parameter-based labels for better grouping
        param_based_labels = self.create_parameter_based_labels(parameters)
        
        print(f"Loaded {len(top_images)} image pairs from {len(set(original_labels))} original folders")
        print(f"Grouped into {len(set(param_based_labels))} parameter-based classes")
        
        # Store data statistics for analysis
        if len(parameters) > 0:
            params_array = np.array(parameters)
            from collections import Counter
            label_distribution = Counter(param_based_labels)
            
            self._data_stats = {
                'total_samples': len(top_images),
                'original_folders': len(set(original_labels)),
                'parameter_based_classes': len(set(param_based_labels)),
                'samples_per_class': dict(label_distribution),
                'parameter_ranges': {
                    '弯曲强度': (np.min(params_array[:, 0]), np.max(params_array[:, 0])),
                    '强度': (np.min(params_array[:, 1]), np.max(params_array[:, 1])),
                    '形变强度': (np.min(params_array[:, 2]), np.max(params_array[:, 2])),
                    '形变率': (np.min(params_array[:, 3]), np.max(params_array[:, 3]))
                },
                'parameter_means': np.mean(params_array, axis=0),
                'parameter_stds': np.std(params_array, axis=0)
            }
            
            print("Parameter Statistics:")
            for i, name in enumerate(['弯曲强度', '强度', '形变强度', '形变率']):
                print(f"  {name}: range {self._data_stats['parameter_ranges'][name]}, "
                      f"mean {self._data_stats['parameter_means'][i]:.2f}")
            
            print("\nClass Distribution (top 10):")
            for label, count in sorted(label_distribution.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {label}: {count} samples")
        
        return (np.array(top_images), np.array(side_images), 
                np.array(parameters), np.array(param_based_labels), label_names)
    
    def compute_parameter_similarity(self, parameters):
        """Compute similarity matrix based on physical parameters"""
        # Normalize parameters to [0,1] range for fair comparison
        params_array = np.array(parameters)
        
        # Use min-max normalization to handle different parameter scales
        params_min = np.min(params_array, axis=0)
        params_max = np.max(params_array, axis=0)
        
        # Avoid division by zero
        params_range = params_max - params_min
        params_range[params_range == 0] = 1.0
        
        params_normalized = (params_array - params_min) / params_range
        
        # Apply parameter importance weights from patent findings
        weighted_params = params_normalized * self.parameter_weights
        
        # Compute cosine similarity matrix
        similarity_matrix = cosine_similarity(weighted_params)
        
        return similarity_matrix
    
    def get_data_statistics(self):
        """Get statistics about the loaded data"""
        if not hasattr(self, '_data_stats'):
            print("Please load data first using load_dual_view_data()")
            return None
            
        return self._data_stats

def create_patent_multimodal_model(num_classes, input_shape=(224, 224, 3)):
    """
    Create multi-modal model following patent methodology:
    - Dual-view image inputs (top and side)
    - Physical parameter inputs with importance weighting
    - Multi-modal fusion architecture
    """
    
    # Image inputs
    top_view = Input(shape=input_shape, name='top_view')
    side_view = Input(shape=input_shape, name='side_view')
    
    # Physical parameters input [弯曲强度, 强度, 形变强度, 形变率]
    physical_params = Input(shape=(4,), name='physical_params')
    
    # Shared CNN backbone for feature extraction
    base_cnn = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_cnn.trainable = False  # Start with frozen weights
    
    # Extract features from both views
    top_features = GlobalAveragePooling2D(name='top_gap')(base_cnn(top_view))
    side_features = GlobalAveragePooling2D(name='side_gap')(base_cnn(side_view))
    
    # Combine visual features
    visual_features = Concatenate(name='visual_concat')([top_features, side_features])
    visual_dense = Dense(512, activation='relu', name='visual_dense1')(visual_features)
    visual_dense = BatchNormalization(name='visual_bn1')(visual_dense)
    visual_dense = Dropout(0.3, name='visual_dropout1')(visual_dense)
    visual_dense = Dense(256, activation='relu', name='visual_dense2')(visual_dense)
    visual_dense = BatchNormalization(name='visual_bn2')(visual_dense)
    visual_dense = Dropout(0.2, name='visual_dropout2')(visual_dense)
    
    # Process physical parameters with importance weighting
    # Based on patent: 弯曲强度(最显著), 强度(较显著), 变形强度(较显著), 变形率(不显著)
    
    # Apply parameter normalization and scaling
    param_normalized = BatchNormalization(name='param_normalization')(physical_params)
    
    # Dense layers for parameter processing with implicit weighting
    param_dense = Dense(128, activation='relu', name='param_dense1')(param_normalized)
    param_dense = BatchNormalization(name='param_bn1')(param_dense)
    param_dense = Dropout(0.2, name='param_dropout1')(param_dense)
    param_dense = Dense(64, activation='relu', name='param_dense2')(param_dense)
    param_dense = BatchNormalization(name='param_bn2')(param_dense)
    
    # Multi-modal fusion with attention mechanism
    combined_features = Concatenate(name='multimodal_fusion')([visual_dense, param_dense])
    
    # Attention-like weighting for multi-modal fusion
    attention_weights = Dense(visual_dense.shape[-1] + param_dense.shape[-1], 
                             activation='softmax', name='fusion_attention')(combined_features)
    attended_features = Multiply(name='attended_fusion')([combined_features, attention_weights])
    
    # Fusion layers
    combined = Dense(512, activation='relu', name='fusion_dense1')(attended_features)
    combined = BatchNormalization(name='fusion_bn1')(combined)
    combined = Dropout(0.4, name='fusion_dropout1')(combined)
    combined = Dense(256, activation='relu', name='fusion_dense2')(combined)
    combined = BatchNormalization(name='fusion_bn2')(combined)
    combined = Dropout(0.3, name='fusion_dropout2')(combined)
    combined = Dense(128, activation='relu', name='fusion_dense3')(combined)
    combined = BatchNormalization(name='fusion_bn3')(combined)
    combined = Dropout(0.2, name='fusion_dropout3')(combined)
    
    # Output layer
    output = Dense(num_classes, activation='softmax', name='classification_output')(combined)
    
    # Create model
    model = Model(inputs=[top_view, side_view, physical_params], 
                  outputs=output, name='PatentBasedMaterialModel')
    
    return model

def get_model_summary(model):
    """Get comprehensive model summary with layer details"""
    print("="*80)
    print("Patent-Based Multi-Modal Model Architecture Summary")
    print("="*80)
    
    # Basic info
    print(f"Total Parameters: {model.count_params():,}")
    print(f"Trainable Parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
    print(f"Non-trainable Parameters: {sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights]):,}")
    
    # Input/Output info
    print(f"\nModel Inputs:")
    for i, inp in enumerate(model.inputs):
        print(f"  {i+1}. {inp.name}: {inp.shape}")
    
    print(f"\nModel Output:")
    print(f"  Shape: {model.output.shape}")
    print(f"  Classes: {model.output.shape[-1]}")
    
    # Architecture breakdown
    print(f"\nArchitecture Breakdown:")
    print("  1. Visual Processing:")
    print("     - ResNet50 backbone (frozen initially)")
    print("     - Dual-view processing (top + side)")
    print("     - Global Average Pooling")
    print("     - Dense layers: 512 -> 256")
    
    print("  2. Parameter Processing:")
    print("     - BatchNormalization")
    print("     - Dense layers: 128 -> 64") 
    print("     - Patent-based importance weighting")
    
    print("  3. Multi-Modal Fusion:")
    print("     - Attention mechanism")
    print("     - Dense layers: 512 -> 256 -> 128")
    print("     - Dropout regularization")
    
    print("  4. Classification:")
    print("     - Softmax output")
    print("     - Parameter-based class grouping")
    
    return model

def test_model_with_real_data(model, data_loader):
    """Test model with actual dataset"""
    print("\n" + "="*60)
    print("Testing Model with Real Data")
    print("="*60)
    
    # Load data
    top_images, side_images, parameters, labels, label_names = data_loader.load_dual_view_data()
    
    # Test with first few samples
    test_size = min(5, len(top_images))
    test_top = top_images[:test_size]
    test_side = side_images[:test_size]
    test_params = parameters[:test_size]
    test_labels = labels[:test_size]
    
    print(f"Testing with {test_size} real samples...")
    
    # Forward pass
    predictions = model.predict([test_top, test_side, test_params], verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)
    
    # Show results
    print("\nPrediction Results:")
    for i in range(test_size):
        print(f"Sample {i+1}:")
        print(f"  True Label: {test_labels[i]}")
        print(f"  Predicted Class: {predicted_classes[i]}")
        print(f"  Confidence: {confidences[i]:.4f}")
        print(f"  Parameters: {test_params[i]}")
        print()
    
    return predictions

def physics_informed_loss(parameter_similarity_matrix, alpha=0.15):
    """
    Create physics-informed loss function that considers material parameter relationships
    """
    def loss_function(y_true, y_pred):
        # Standard classification loss
        classification_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        
        # Physics consistency loss - materials with similar parameters should have similar predictions
        param_consistency_loss = tf.reduce_mean(
            tf.square(y_pred - tf.matmul(parameter_similarity_matrix, y_pred))
        )
        
        # Combined loss
        total_loss = classification_loss + alpha * param_consistency_loss
        
        return total_loss
    
    return loss_function

def create_data_augmentation():
    """Create patent-compliant data augmentation"""
    return ImageDataGenerator(
        rotation_range=10,  # Limited rotation to maintain fabric orientation
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        brightness_range=[0.9, 1.1],  # Simulate lighting conditions
        horizontal_flip=False,  # Don't flip - fabric orientation matters
        fill_mode='nearest'
    )

if __name__ == "__main__":
    print("Patent-Based Material Recognition Model")
    print("Based on: 一种基于AI图形识别技术的数字面料物理属性获取方法")
    
    # Initialize data loader
    data_loader = PatentDataLoader('./Materials_data')
    
    # Load dual-view data
    top_images, side_images, parameters, labels, label_names = data_loader.load_dual_view_data()
    
    if len(top_images) == 0:
        print("No valid image pairs found. Please check your data structure.")
        exit()
    
    print(f"Dataset loaded: {len(top_images)} samples, {len(np.unique(labels))} classes")
    print("Sample parameters:", parameters[:5])
    print("Sample labels:", labels[:5])