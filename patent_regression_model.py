"""
Patent-Based Material Parameter Regression Model
Based on: 一种基于AI图形识别技术的数字面料物理属性获取方法

Correct Implementation:
- REGRESSION task: Predict continuous parameter values from dual-view images
- Input: Top-view + Side-view images (30cm×30cm fabric on standard platform)
- Output: 4 continuous parameters [弯曲强度, 强度, 形变强度, 形变率]
- Goal: Replace expensive physical testing equipment with AI image analysis
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Concatenate, Dropout, 
                                   GlobalAveragePooling2D, BatchNormalization,
                                   Lambda, Add, Multiply)
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
# import seaborn as sns  # Optional for correlation matrix visualization
import imghdr
from collections import Counter

class PatentRegressionDataLoader:
    """
    Regression data loader following patent specifications:
    - Predicts continuous parameter values from dual-view images
    - Target: [弯曲强度, 强度, 形变强度, 形变率] (continuous values 0-100)
    - Patent methodology: Replace physical testing with AI image analysis
    """
    
    def __init__(self, data_path, image_size=(224, 224), normalize_params=True):
        self.data_path = data_path
        self.image_size = image_size
        self.normalize_params = normalize_params
        
        # Parameter importance weights from patent findings
        # 弯曲强度(最显著), 强度(较显著), 形变强度(较显著), 形变率(不显著)
        self.parameter_weights = np.array([1.0, 0.6, 0.6, 0.3], dtype=np.float32)
        
        # Parameter names for reference
        self.param_names = ['弯曲强度', '强度', '形变强度', '形变率']
        self.param_names_en = ['Bending_Strength', 'Strength', 'Deformation_Strength', 'Deformation_Rate']
        
        # Parameter scalers for normalization
        self.param_scaler = None
        
    def extract_parameters_from_folder(self, folder_name):
        """
        Extract continuous physical parameters from folder name for regression
        Format: "1-20.20.20.20-1" -> [20.0, 20.0, 20.0, 20.0]
        Returns raw parameter values (not discretized)
        """
        try:
            parts = folder_name.split('-')
            if len(parts) >= 2:
                param_str = parts[1]  # "20.20.20.20"
                params = [float(x) for x in param_str.split('.')]
                if len(params) >= 4:
                    # Validate parameter ranges and return raw values
                    validated_params = []
                    for i, param in enumerate(params[:4]):
                        # Ensure parameters are in valid range [0, 100]
                        validated_params.append(max(0.0, min(100.0, param)))
                    return validated_params
        except Exception as e:
            print(f"Error extracting parameters from {folder_name}: {e}")
        
        # Default parameters if extraction fails
        return [0.0, 20.0, 20.0, 20.0]
    
    def load_and_preprocess_image(self, image_path):
        """Load and preprocess image according to patent specifications"""
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
    
    def load_regression_data(self):
        """
        Load dual-view images with continuous parameter targets for regression
        Returns: top_images, side_images, parameter_values, folder_names
        """
        top_images = []
        side_images = []
        parameter_values = []
        folder_names = []
        image_paths = []
        
        print("Loading patent-based regression data...")
        print("Target: Continuous parameter prediction [弯曲强度, 强度, 形变强度, 形变率]")
        
        for folder in sorted(os.listdir(self.data_path)):
            folder_path = os.path.join(self.data_path, folder)
            
            if not os.path.isdir(folder_path):
                continue
                
            # Extract continuous physical parameters from folder name
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
            
            # Load valid pairs for regression training
            for base_name, pair in image_pairs.items():
                if 'top' in pair and 'side' in pair:
                    top_path = os.path.join(folder_path, pair['top'])
                    side_path = os.path.join(folder_path, pair['side'])
                    
                    top_img = self.load_and_preprocess_image(top_path)
                    side_img = self.load_and_preprocess_image(side_path)
                    
                    if top_img is not None and side_img is not None:
                        top_images.append(top_img)
                        side_images.append(side_img)
                        parameter_values.append(params)
                        folder_names.append(folder)
                        image_paths.append((top_path, side_path))
        
        # Convert to numpy arrays
        top_images = np.array(top_images)
        side_images = np.array(side_images)
        parameter_values = np.array(parameter_values, dtype=np.float32)
        
        print(f"Loaded {len(top_images)} image pairs for regression")
        print(f"Parameter shape: {parameter_values.shape}")
        
        # Store data statistics for analysis
        self._store_regression_statistics(parameter_values, folder_names)
        
        # Normalize parameters if requested
        if self.normalize_params:
            parameter_values = self._normalize_parameters(parameter_values)
            print("Parameters normalized to [0,1] range")
        
        return top_images, side_images, parameter_values, folder_names, image_paths
    
    def _normalize_parameters(self, parameter_values):
        """Normalize parameters to [0,1] range for better training stability"""
        if self.param_scaler is None:
            # Use MinMaxScaler to normalize to [0,1] range
            self.param_scaler = MinMaxScaler()
            normalized_params = self.param_scaler.fit_transform(parameter_values)
        else:
            normalized_params = self.param_scaler.transform(parameter_values)
        
        return normalized_params.astype(np.float32)
    
    def denormalize_parameters(self, normalized_params):
        """Convert normalized parameters back to original scale"""
        if self.param_scaler is None:
            print("Warning: No scaler available for denormalization")
            return normalized_params
        
        return self.param_scaler.inverse_transform(normalized_params)
    
    def _store_regression_statistics(self, parameter_values, folder_names):
        """Store regression data statistics for analysis"""
        self._data_stats = {
            'total_samples': len(parameter_values),
            'unique_folders': len(set(folder_names)),
            'parameter_statistics': {}
        }
        
        print("\nRegression Target Statistics:")
        for i, (name_zh, name_en) in enumerate(zip(self.param_names, self.param_names_en)):
            param_vals = parameter_values[:, i]
            stats = {
                'mean': np.mean(param_vals),
                'std': np.std(param_vals),
                'min': np.min(param_vals),
                'max': np.max(param_vals),
                'median': np.median(param_vals)
            }
            self._data_stats['parameter_statistics'][name_en] = stats
            
            print(f"  {name_zh} ({name_en}):")
            print(f"    Range: [{stats['min']:.1f}, {stats['max']:.1f}]")
            print(f"    Mean±Std: {stats['mean']:.1f}±{stats['std']:.1f}")
        
        # Check parameter distribution
        print(f"\nParameter Distribution Analysis:")
        for i, name in enumerate(self.param_names_en):
            unique_vals = len(np.unique(parameter_values[:, i]))
            print(f"  {name}: {unique_vals} unique values")
    
    def visualize_parameter_distributions(self, parameter_values, save_path='parameter_distributions.png'):
        """Visualize parameter distributions for regression analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, (name_zh, name_en) in enumerate(zip(self.param_names, self.param_names_en)):
            ax = axes[i]
            param_vals = parameter_values[:, i]
            
            # Histogram
            ax.hist(param_vals, bins=20, alpha=0.7, edgecolor='black', color=f'C{i}')
            ax.set_xlabel(f'{name_zh} ({name_en})')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{name_zh} Distribution\nRange: [{np.min(param_vals):.1f}, {np.max(param_vals):.1f}]')
            ax.grid(True, alpha=0.3)
            
            # Add statistics text
            mean_val = np.mean(param_vals)
            std_val = np.std(param_vals)
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.1f}')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Parameter distributions saved to {save_path}")
    
    def get_parameter_correlation_matrix(self, parameter_values):
        """Compute correlation matrix between parameters"""
        import pandas as pd
        
        df = pd.DataFrame(parameter_values, columns=self.param_names_en)
        correlation_matrix = df.corr()
        
        # Visualize correlation matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar()
        plt.title('Parameter Correlation Matrix')
        
        # Add correlation values as text
        for i in range(len(correlation_matrix)):
            for j in range(len(correlation_matrix)):
                plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                        ha='center', va='center')
        
        plt.xticks(range(len(self.param_names)), self.param_names, rotation=45)
        plt.yticks(range(len(self.param_names)), self.param_names)
        plt.tight_layout()
        plt.savefig('parameter_correlations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return correlation_matrix

def create_patent_regression_model(input_shape=(224, 224, 3)):
    """
    Create patent-based regression model for continuous parameter prediction
    
    Architecture:
    - Dual-view CNN feature extraction (ResNet50 backbone)
    - Multi-modal feature fusion
    - Regression head for 4 continuous parameters
    """
    
    # Dual-view image inputs
    top_view = Input(shape=input_shape, name='top_view')
    side_view = Input(shape=input_shape, name='side_view')
    
    # Shared CNN backbone for feature extraction
    base_cnn = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_cnn.trainable = False  # Start with frozen weights for transfer learning
    
    # Extract features from both views
    top_features = base_cnn(top_view)
    side_features = base_cnn(side_view)
    
    # Global Average Pooling
    top_features = GlobalAveragePooling2D(name='top_gap')(top_features)
    side_features = GlobalAveragePooling2D(name='side_gap')(side_features)
    
    # View-specific processing
    top_processed = Dense(512, activation='relu', name='top_dense')(top_features)
    top_processed = BatchNormalization(name='top_bn')(top_processed)
    top_processed = Dropout(0.3, name='top_dropout')(top_processed)
    
    side_processed = Dense(512, activation='relu', name='side_dense')(side_features)
    side_processed = BatchNormalization(name='side_bn')(side_processed)
    side_processed = Dropout(0.3, name='side_dropout')(side_processed)
    
    # Multi-view fusion with attention
    combined_features = Concatenate(name='view_fusion')([top_processed, side_processed])
    
    # Attention mechanism for view importance
    attention_weights = Dense(1024, activation='softmax', name='view_attention')(combined_features)
    attended_features = Multiply(name='attended_features')([combined_features, attention_weights])
    
    # Regression head with parameter-specific branches
    fusion_dense = Dense(512, activation='relu', name='fusion_dense1')(attended_features)
    fusion_dense = BatchNormalization(name='fusion_bn1')(fusion_dense)
    fusion_dense = Dropout(0.4, name='fusion_dropout1')(fusion_dense)
    
    fusion_dense = Dense(256, activation='relu', name='fusion_dense2')(fusion_dense)
    fusion_dense = BatchNormalization(name='fusion_bn2')(fusion_dense)
    fusion_dense = Dropout(0.3, name='fusion_dropout2')(fusion_dense)
    
    # Final regression output - 4 continuous parameters
    # Using linear activation for regression (no bounds imposed here)
    parameter_predictions = Dense(4, activation='linear', name='parameter_regression')(fusion_dense)
    
    # Create model
    model = Model(inputs=[top_view, side_view], outputs=parameter_predictions, 
                  name='PatentRegressionModel')
    
    return model

def parameter_weighted_mse_loss(parameter_weights):
    """
    Create parameter-weighted MSE loss based on patent importance findings
    
    Args:
        parameter_weights: Array of importance weights [弯曲强度, 强度, 形变强度, 形变率]
    """
    def weighted_mse(y_true, y_pred):
        # Compute squared errors for each parameter
        squared_errors = tf.square(y_true - y_pred)
        
        # Apply importance weights from patent
        weighted_errors = squared_errors * tf.constant(parameter_weights, dtype=tf.float32)
        
        # Return mean weighted error
        return tf.reduce_mean(weighted_errors)
    
    return weighted_mse

def parameter_weighted_mae_loss(parameter_weights):
    """
    Create parameter-weighted MAE loss for more robust training
    """
    def weighted_mae(y_true, y_pred):
        # Compute absolute errors for each parameter
        abs_errors = tf.abs(y_true - y_pred)
        
        # Apply importance weights from patent
        weighted_errors = abs_errors * tf.constant(parameter_weights, dtype=tf.float32)
        
        # Return mean weighted error
        return tf.reduce_mean(weighted_errors)
    
    return weighted_mae

if __name__ == "__main__":
    print("="*80)
    print("Patent-Based Material Parameter Regression")
    print("Target: Continuous parameter prediction from dual-view images")
    print("="*80)
    
    # Initialize regression data loader
    data_loader = PatentRegressionDataLoader('./Materials_data', normalize_params=True)
    
    # Load regression data
    top_images, side_images, parameter_values, folder_names, image_paths = data_loader.load_regression_data()
    
    # Visualize parameter distributions
    if len(parameter_values) > 0:
        print("\nVisualizing parameter distributions...")
        data_loader.visualize_parameter_distributions(parameter_values)
        
        print("\nComputing parameter correlations...")
        correlation_matrix = data_loader.get_parameter_correlation_matrix(parameter_values)
        
        print("\nRegression data loading completed successfully!")
        print(f"Ready for training with {len(top_images)} image pairs")
    else:
        print("ERROR: No valid regression data found!")