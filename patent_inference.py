"""
Patent-Based Material Recognition Inference Script
Implements material similarity detection following patent methodology
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt
from patent_based_model import PatentDataLoader

class PatentMaterialPredictor:
    """
    Material recognition predictor following patent methodology
    """
    
    def __init__(self, model_path, data_path='./Materials_data'):
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.label_encoder = None
        self.reference_data = None
        
        # Parameter names in Chinese and English
        self.param_names = {
            0: '弯曲强度 (Bending Strength)',
            1: '强度 (Strength)', 
            2: '形变强度 (Deformation Strength)',
            3: '形变率 (Deformation Rate)'
        }
        
        # Parameter importance weights from patent
        self.param_weights = np.array([1.0, 0.6, 0.6, 0.3])
        
    def load_model_and_data(self):
        """Load trained model and reference dataset"""
        print("Loading patent-based model and reference data...")
        
        try:
            # Load model
            self.model = load_model(self.model_path)
            print(f"✓ Model loaded from {self.model_path}")
            
            # Load reference data
            data_loader = PatentDataLoader(self.data_path)
            top_images, side_images, parameters, labels, label_names = data_loader.load_dual_view_data()
            
            # Encode labels
            self.label_encoder = LabelEncoder()
            encoded_labels = self.label_encoder.fit_transform(labels)
            
            self.reference_data = {
                'top_images': top_images,
                'side_images': side_images,
                'parameters': parameters,
                'labels': labels,
                'encoded_labels': encoded_labels,
                'label_names': label_names
            }
            
            print(f"✓ Reference dataset loaded: {len(top_images)} samples, {len(np.unique(labels))} classes")
            
        except Exception as e:
            print(f"Error loading model or data: {e}")
            raise
    
    def preprocess_input_images(self, top_image_path, side_image_path):
        """Preprocess input image pair for prediction"""
        try:
            # Load and preprocess top view
            top_img = load_img(top_image_path, target_size=(224, 224), color_mode='rgb')
            top_array = img_to_array(top_img) / 255.0
            
            # Load and preprocess side view
            side_img = load_img(side_image_path, target_size=(224, 224), color_mode='rgb')
            side_array = img_to_array(side_img) / 255.0
            
            return top_array, side_array
            
        except Exception as e:
            print(f"Error preprocessing images: {e}")
            return None, None
    
    def predict_material_properties(self, top_image_path, side_image_path, known_params=None):
        """
        Predict material properties using patent-based multi-modal approach
        
        Args:
            top_image_path: Path to top-view image
            side_image_path: Path to side-view image  
            known_params: Known physical parameters [弯曲强度, 强度, 形变强度, 形变率]
                         If None, will estimate from visual features
        """
        
        if self.model is None:
            self.load_model_and_data()
        
        # Preprocess images
        top_img, side_img = self.preprocess_input_images(top_image_path, side_image_path)
        if top_img is None or side_img is None:
            return None
        
        # Prepare input arrays
        top_input = np.expand_dims(top_img, axis=0)
        side_input = np.expand_dims(side_img, axis=0)
        
        # Handle parameters
        if known_params is not None:
            # Use provided parameters
            param_input = np.array([known_params], dtype=np.float32)
        else:
            # Estimate parameters from visual similarity (fallback)
            param_input = self._estimate_parameters_from_visual(top_img, side_img)
        
        # Make prediction
        try:
            predictions = self.model.predict([top_input, side_input, param_input])
            pred_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            # Get material name
            material_name = self.label_encoder.classes_[pred_class]
            
            return {
                'predicted_class': material_name,
                'confidence': confidence,
                'class_probabilities': predictions[0],
                'used_parameters': param_input[0],
                'top_image_path': top_image_path,
                'side_image_path': side_image_path
            }
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None
    
    def _estimate_parameters_from_visual(self, top_img, side_img):
        """Estimate physical parameters from visual similarity to reference data"""
        # This is a fallback method when parameters are unknown
        # In practice, the patent methodology assumes parameters are provided
        
        # Use average parameters as fallback
        avg_params = np.mean(self.reference_data['parameters'], axis=0)
        return np.array([avg_params], dtype=np.float32)
    
    def find_similar_materials(self, top_image_path, side_image_path, known_params, top_k=5):
        """
        Find top-k most similar materials using patent methodology
        Combines visual and parameter similarity
        """
        
        if self.model is None:
            self.load_model_and_data()
        
        # Get prediction
        prediction_result = self.predict_material_properties(top_image_path, side_image_path, known_params)
        if prediction_result is None:
            return None
        
        # Extract features using the model (remove final classification layer)
        feature_extractor = tf.keras.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer('multimodal_fusion').output
        )
        
        # Get query features
        top_img, side_img = self.preprocess_input_images(top_image_path, side_image_path)
        top_input = np.expand_dims(top_img, axis=0)
        side_input = np.expand_dims(side_img, axis=0)
        param_input = np.array([known_params], dtype=np.float32)
        
        query_features = feature_extractor.predict([top_input, side_input, param_input])
        
        # Get reference features
        ref_features = feature_extractor.predict([
            self.reference_data['top_images'],
            self.reference_data['side_images'], 
            self.reference_data['parameters']
        ])
        
        # Compute similarities
        visual_similarities = cosine_similarity(query_features, ref_features)[0]
        
        # Parameter similarities
        param_similarities = cosine_similarity([known_params], self.reference_data['parameters'])[0]
        
        # Combined similarity (visual + parameter weighted)
        combined_similarities = 0.7 * visual_similarities + 0.3 * param_similarities
        
        # Get top-k similar materials
        top_indices = np.argsort(combined_similarities)[::-1][:top_k]
        
        results = []
        for i, idx in enumerate(top_indices):
            results.append({
                'rank': i + 1,
                'material_name': self.reference_data['labels'][idx],
                'visual_similarity': visual_similarities[idx],
                'parameter_similarity': param_similarities[idx],
                'combined_similarity': combined_similarities[idx],
                'reference_parameters': self.reference_data['parameters'][idx],
                'parameter_diff': np.abs(np.array(known_params) - self.reference_data['parameters'][idx])
            })
        
        return {
            'query_prediction': prediction_result,
            'query_parameters': known_params,
            'similar_materials': results
        }
    
    def format_material_details(self, material_name, parameters):
        """Format material details following patent format"""
        # Parse material name (e.g., "1-20.20.20.20-1")
        try:
            parts = material_name.split('-')
            material_id = parts[0]
            
            details = f"材料{material_id}"
            
            # Add parameter details
            for i, (param_val, param_name) in enumerate(zip(parameters, self.param_names.values())):
                details += f", {param_name}: {param_val:.2f}"
            
            return details
            
        except:
            return f"材料: {material_name}, 参数: {parameters}"
    
    def print_similarity_results(self, results, show_details=True):
        """Print similarity results in a formatted way"""
        if results is None:
            print("No results to display")
            return
        
        print("="*80)
        print("Patent-Based Material Recognition Results")
        print("="*80)
        
        # Query information
        query_pred = results['query_prediction']
        print(f"Query Image: {os.path.basename(query_pred['top_image_path'])} (top), {os.path.basename(query_pred['side_image_path'])} (side)")
        print(f"Predicted Material: {query_pred['predicted_class']}")
        print(f"Prediction Confidence: {query_pred['confidence']:.4f}")
        print(f"Query Parameters: {results['query_parameters']}")
        
        print("\n" + "-"*80)
        print("Most Similar Materials:")
        print("-"*80)
        
        for material in results['similar_materials']:
            print(f"{material['rank']}. {self.format_material_details(material['material_name'], material['reference_parameters'])}")
            print(f"   Combined Similarity: {material['combined_similarity']:.4f}")
            print(f"   Visual Similarity: {material['visual_similarity']:.4f}")  
            print(f"   Parameter Similarity: {material['parameter_similarity']:.4f}")
            
            if show_details:
                print(f"   Parameter Differences: {material['parameter_diff']}")
            print()

# Example usage functions
def test_material_recognition():
    """Test the patent-based material recognition system"""
    
    # Initialize predictor
    predictor = PatentMaterialPredictor('patent_based_material_model.h5')
    
    # Test with sample images (you need to provide actual paths)
    test_cases = [
        {
            'top_image': './testing/sample_top.jpg',
            'side_image': './testing/sample_side.jpg', 
            'parameters': [30.0, 25.0, 25.0, 25.0],  # [弯曲强度, 强度, 形变强度, 形变率]
            'description': 'Test fabric sample 1'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test Case {i}: {test_case['description']}")
        print(f"{'='*60}")
        
        if not os.path.exists(test_case['top_image']) or not os.path.exists(test_case['side_image']):
            print(f"Test images not found, skipping test case {i}")
            continue
        
        # Find similar materials
        results = predictor.find_similar_materials(
            test_case['top_image'],
            test_case['side_image'], 
            test_case['parameters'],
            top_k=3
        )
        
        # Display results
        predictor.print_similarity_results(results)

if __name__ == "__main__":
    # Run test
    test_material_recognition()