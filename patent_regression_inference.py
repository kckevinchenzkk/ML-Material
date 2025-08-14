"""
Patent-Based Material Parameter Regression Inference
Implements continuous parameter prediction and similarity computation

Based on: 一种基于AI图形识别技术的数字面料物理属性获取方法
Goal: Predict continuous parameters and find similar materials
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
from patent_regression_model import PatentRegressionDataLoader, parameter_weighted_mse_loss, parameter_weighted_mae_loss

class PatentRegressionPredictor:
    """
    Material parameter regression predictor following patent methodology
    Predicts continuous parameter values and finds similar materials
    """
    
    def __init__(self, model_path, data_path='./Materials_data'):
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.reference_data = None
        self.param_scaler = None
        
        # Parameter names and weights from patent
        self.param_names = ['弯曲强度', '强度', '形变强度', '形变率']
        self.param_names_en = ['Bending_Strength', 'Strength', 'Deformation_Strength', 'Deformation_Rate']
        self.param_weights = np.array([1.0, 0.6, 0.6, 0.3])  # Patent importance weights
        
    def load_model_and_data(self):
        """Load trained regression model and reference dataset"""
        print("Loading patent-based regression model and reference data...")
        
        try:
            # Load model with custom loss functions
            parameter_weights = self.param_weights
            custom_objects = {
                'weighted_mse': parameter_weighted_mse_loss(parameter_weights),
                'weighted_mae': parameter_weighted_mae_loss(parameter_weights)
            }
            
            self.model = load_model(self.model_path, custom_objects=custom_objects)
            print(f"Model loaded from {self.model_path}")
            
            # Load reference data
            data_loader = PatentRegressionDataLoader(self.data_path, normalize_params=True)
            top_images, side_images, parameter_values, folder_names, image_paths = data_loader.load_regression_data()
            
            self.param_scaler = data_loader.param_scaler
            
            self.reference_data = {
                'top_images': top_images,
                'side_images': side_images,
                'parameters_normalized': parameter_values,
                'folder_names': folder_names,
                'image_paths': image_paths
            }
            
            # Store original parameter values for comparison
            if self.param_scaler is not None:
                self.reference_data['parameters_original'] = data_loader.param_scaler.inverse_transform(parameter_values)
            else:
                self.reference_data['parameters_original'] = parameter_values
            
            print(f"Reference dataset loaded: {len(top_images)} samples")
            
        except Exception as e:
            print(f"Error loading model or data: {e}")
            raise
    
    def preprocess_input_images(self, top_image_path, side_image_path):
        """Preprocess input image pair for regression prediction"""
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
    
    def predict_material_parameters(self, top_image_path, side_image_path):
        """
        Predict continuous material parameters from dual-view images
        
        Args:
            top_image_path: Path to top-view image
            side_image_path: Path to side-view image
            
        Returns:
            Dictionary with predicted parameters and metadata
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
        
        # Make prediction
        try:
            predictions_normalized = self.model.predict([top_input, side_input], verbose=0)
            
            # Denormalize predictions to original scale
            if self.param_scaler is not None:
                predictions_original = self.param_scaler.inverse_transform(predictions_normalized)
            else:
                predictions_original = predictions_normalized
            
            # Format results
            result = {
                'predicted_parameters_original': predictions_original[0],
                'predicted_parameters_normalized': predictions_normalized[0],
                'parameter_names': self.param_names,
                'parameter_names_en': self.param_names_en,
                'top_image_path': top_image_path,
                'side_image_path': side_image_path
            }
            
            # Add individual parameter results
            for i, (name_zh, name_en) in enumerate(zip(self.param_names, self.param_names_en)):
                result[f'{name_en}_predicted'] = predictions_original[0][i]
            
            return result
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None
    
    def compute_parameter_similarity(self, predicted_params, reference_params, method='weighted_euclidean'):
        """
        Compute similarity between predicted and reference parameters
        
        Args:
            predicted_params: Predicted parameter values
            reference_params: Reference parameter matrix
            method: Similarity method ('weighted_euclidean', 'cosine', 'euclidean')
        """
        
        if method == 'weighted_euclidean':
            # Use patent-based parameter importance weights
            weighted_pred = predicted_params * self.param_weights
            weighted_ref = reference_params * self.param_weights.reshape(1, -1)
            
            # Compute weighted Euclidean distances
            distances = np.sqrt(np.sum((weighted_ref - weighted_pred.reshape(1, -1))**2, axis=1))
            similarities = 1.0 / (1.0 + distances)  # Convert distances to similarities
            
        elif method == 'cosine':
            # Cosine similarity
            similarities = cosine_similarity([predicted_params], reference_params)[0]
            
        elif method == 'euclidean':
            # Standard Euclidean distances
            distances = np.sqrt(np.sum((reference_params - predicted_params.reshape(1, -1))**2, axis=1))
            similarities = 1.0 / (1.0 + distances)
            
        else:
            raise ValueError(f"Unknown similarity method: {method}")
        
        return similarities
    
    def find_similar_materials(self, top_image_path, side_image_path, top_k=5, similarity_method='weighted_euclidean'):
        """
        Find materials with similar predicted parameters
        
        Args:
            top_image_path: Path to top-view image
            side_image_path: Path to side-view image
            top_k: Number of most similar materials to return
            similarity_method: Method for computing similarity
        """
        
        if self.model is None:
            self.load_model_and_data()
        
        # Get parameter prediction
        prediction_result = self.predict_material_parameters(top_image_path, side_image_path)
        if prediction_result is None:
            return None
        
        # Compute similarities using original scale parameters
        predicted_params = prediction_result['predicted_parameters_original']
        reference_params = self.reference_data['parameters_original']
        
        similarities = self.compute_parameter_similarity(
            predicted_params, 
            reference_params, 
            method=similarity_method
        )
        
        # Get top-k most similar materials
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for i, idx in enumerate(top_indices):
            parameter_diff = np.abs(predicted_params - reference_params[idx])
            weighted_diff = parameter_diff * self.param_weights
            
            results.append({
                'rank': i + 1,
                'material_id': self.reference_data['folder_names'][idx],
                'similarity_score': similarities[idx],
                'reference_parameters': reference_params[idx],
                'parameter_differences': parameter_diff,
                'weighted_differences': weighted_diff,
                'image_paths': self.reference_data['image_paths'][idx]
            })
        
        return {
            'query_prediction': prediction_result,
            'similarity_method': similarity_method,
            'similar_materials': results,
            'parameter_importance_weights': self.param_weights
        }
    
    def format_prediction_results(self, prediction_result):
        """Format prediction results for display"""
        if prediction_result is None:
            return "No prediction results available"
        
        output = []
        output.append("Material Parameter Prediction Results")
        output.append("="*60)
        output.append(f"Top Image: {os.path.basename(prediction_result['top_image_path'])}")
        output.append(f"Side Image: {os.path.basename(prediction_result['side_image_path'])}")
        output.append("")
        output.append("Predicted Parameters:")
        
        for i, (name_zh, name_en) in enumerate(zip(self.param_names, self.param_names_en)):
            value = prediction_result['predicted_parameters_original'][i]
            output.append(f"  {name_zh} ({name_en}): {value:.2f}")
        
        return "\n".join(output)
    
    def format_similarity_results(self, similarity_results, show_details=True):
        """Format similarity results for display"""
        if similarity_results is None:
            return "No similarity results available"
        
        output = []
        output.append("Material Similarity Analysis")
        output.append("="*80)
        
        # Query information
        query = similarity_results['query_prediction']
        output.append(f"Query: {os.path.basename(query['top_image_path'])} (top), {os.path.basename(query['side_image_path'])} (side)")
        output.append(f"Similarity Method: {similarity_results['similarity_method']}")
        output.append("")
        
        output.append("Predicted Parameters:")
        for i, (name_zh, name_en) in enumerate(zip(self.param_names, self.param_names_en)):
            value = query['predicted_parameters_original'][i]
            weight = similarity_results['parameter_importance_weights'][i]
            output.append(f"  {name_zh}: {value:.2f} (weight: {weight:.1f})")
        
        output.append("\n" + "-"*80)
        output.append("Most Similar Materials:")
        output.append("-"*80)
        
        for material in similarity_results['similar_materials']:
            output.append(f"{material['rank']}. Material: {material['material_id']}")
            output.append(f"   Similarity Score: {material['similarity_score']:.4f}")
            
            if show_details:
                output.append("   Reference Parameters:")
                for i, (name_zh, name_en) in enumerate(zip(self.param_names, self.param_names_en)):
                    ref_val = material['reference_parameters'][i]
                    diff = material['parameter_differences'][i]
                    output.append(f"     {name_zh}: {ref_val:.2f} (diff: ±{diff:.2f})")
            
            output.append("")
        
        return "\n".join(output)
    
    def plot_similarity_comparison(self, similarity_results, save_path='similarity_comparison.png'):
        """Plot parameter comparison between query and similar materials"""
        
        if similarity_results is None:
            return
        
        query = similarity_results['query_prediction']
        similar_materials = similarity_results['similar_materials'][:3]  # Top 3
        
        # Prepare data
        materials = ['Query'] + [f"Rank {m['rank']}" for m in similar_materials]
        parameters_data = [query['predicted_parameters_original']] + [m['reference_parameters'] for m in similar_materials]
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        # Parameter angles
        angles = np.linspace(0, 2 * np.pi, len(self.param_names), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Plot each material
        colors = ['red', 'blue', 'green', 'orange']
        for i, (material, params) in enumerate(zip(materials, parameters_data)):
            values = params.tolist() + [params[0]]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=material, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        # Customize plot
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(self.param_names, fontsize=10)
        ax.set_ylim(0, 100)  # Assuming parameters range 0-100
        ax.set_title('Parameter Comparison: Query vs Similar Materials', size=14, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Similarity comparison plot saved to {save_path}")

# Example usage functions
def test_regression_inference():
    """Test the regression inference system"""
    
    # Initialize predictor
    model_path = 'patent_regression_model_final.h5'
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train the model first.")
        return
    
    predictor = PatentRegressionPredictor(model_path)
    
    # Test with sample from reference data
    predictor.load_model_and_data()
    
    # Use a reference sample for testing
    sample_idx = 0
    test_top_path, test_side_path = predictor.reference_data['image_paths'][sample_idx]
    true_params = predictor.reference_data['parameters_original'][sample_idx]
    
    print(f"Testing with reference sample {sample_idx}")
    print(f"True parameters: {true_params}")
    print(f"Images: {test_top_path}, {test_side_path}")
    
    # Test parameter prediction
    print("\n" + "="*60)
    print("Testing Parameter Prediction")
    print("="*60)
    
    prediction_result = predictor.predict_material_parameters(test_top_path, test_side_path)
    print(predictor.format_prediction_results(prediction_result))
    
    # Test similarity search
    print("\n" + "="*60)
    print("Testing Similarity Search")
    print("="*60)
    
    similarity_results = predictor.find_similar_materials(
        test_top_path, 
        test_side_path, 
        top_k=5
    )
    
    print(predictor.format_similarity_results(similarity_results))
    
    # Plot comparison
    predictor.plot_similarity_comparison(similarity_results)

if __name__ == "__main__":
    # Run test
    test_regression_inference()