"""
Patent-Based Material Parameter Regression Training Script
Implements continuous parameter prediction from dual-view images

Based on: 一种基于AI图形识别技术的数字面料物理属性获取方法
Target: Predict continuous values [弯曲强度, 强度, 形变强度, 形变率]
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
from patent_regression_model import (PatentRegressionDataLoader, create_patent_regression_model, 
                                   parameter_weighted_mse_loss, parameter_weighted_mae_loss)

def create_regression_callbacks():
    """Create callbacks for regression training"""
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            'best_patent_regression_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

def evaluate_regression_model(model, X_test, y_test, param_scaler=None):
    """Evaluate regression model with comprehensive metrics"""
    
    # Make predictions
    y_pred = model.predict(X_test, verbose=0)
    
    # Denormalize if scaler is provided
    if param_scaler is not None:
        y_test_orig = param_scaler.inverse_transform(y_test)
        y_pred_orig = param_scaler.inverse_transform(y_pred)
    else:
        y_test_orig = y_test
        y_pred_orig = y_pred
    
    print("\nRegression Evaluation Results:")
    print("="*60)
    
    # Parameter names
    param_names = ['Bending_Strength', 'Strength', 'Deformation_Strength', 'Deformation_Rate']
    param_names_zh = ['弯曲强度', '强度', '形变强度', '形变率']
    
    # Overall metrics
    overall_mse = mean_squared_error(y_test, y_pred)
    overall_mae = mean_absolute_error(y_test, y_pred)
    overall_r2 = r2_score(y_test, y_pred)
    
    print(f"Overall Metrics (Normalized):")
    print(f"  MSE: {overall_mse:.4f}")
    print(f"  MAE: {overall_mae:.4f}")
    print(f"  R²: {overall_r2:.4f}")
    
    # Per-parameter metrics
    print(f"\nPer-Parameter Metrics (Original Scale):")
    for i, (name_en, name_zh) in enumerate(zip(param_names, param_names_zh)):
        mse = mean_squared_error(y_test_orig[:, i], y_pred_orig[:, i])
        mae = mean_absolute_error(y_test_orig[:, i], y_pred_orig[:, i])
        r2 = r2_score(y_test_orig[:, i], y_pred_orig[:, i])
        
        print(f"  {name_zh} ({name_en}):")
        print(f"    MSE: {mse:.4f}")
        print(f"    MAE: {mae:.4f}")
        print(f"    R²: {r2:.4f}")
    
    return {
        'overall_mse': overall_mse,
        'overall_mae': overall_mae,
        'overall_r2': overall_r2,
        'predictions': y_pred,
        'predictions_original': y_pred_orig,
        'true_values': y_test,
        'true_values_original': y_test_orig
    }

def plot_regression_results(results, save_path='regression_results.png'):
    """Plot regression prediction results"""
    
    y_true_orig = results['true_values_original']
    y_pred_orig = results['predictions_original']
    
    param_names = ['Bending Strength', 'Strength', 'Deformation Strength', 'Deformation Rate']
    param_names_zh = ['弯曲强度', '强度', '形变强度', '形变率']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, (ax, name_en, name_zh) in enumerate(zip(axes, param_names, param_names_zh)):
        # Scatter plot of true vs predicted
        ax.scatter(y_true_orig[:, i], y_pred_orig[:, i], alpha=0.6, color=f'C{i}')
        
        # Perfect prediction line
        min_val = min(y_true_orig[:, i].min(), y_pred_orig[:, i].min())
        max_val = max(y_true_orig[:, i].max(), y_pred_orig[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        # Calculate R²
        r2 = r2_score(y_true_orig[:, i], y_pred_orig[:, i])
        mae = mean_absolute_error(y_true_orig[:, i], y_pred_orig[:, i])
        
        ax.set_xlabel(f'True {name_en}')
        ax.set_ylabel(f'Predicted {name_en}')
        ax.set_title(f'{name_zh}\nR² = {r2:.3f}, MAE = {mae:.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Regression results plotted and saved to {save_path}")

def train_patent_regression_model():
    """Main training function for patent-based regression"""
    
    print("="*80)
    print("Patent-Based Material Parameter Regression Training")
    print("Target: Continuous parameter prediction from dual-view images")
    print("="*80)
    
    # 1. Load regression data
    print("\nStep 1: Loading patent-based regression data...")
    data_loader = PatentRegressionDataLoader('./Materials_data', normalize_params=True)
    
    top_images, side_images, parameter_values, folder_names, image_paths = data_loader.load_regression_data()
    
    if len(top_images) == 0:
        print("ERROR: No valid regression data found!")
        return None
    
    print(f"Loaded {len(top_images)} image pairs for regression training")
    print(f"Parameter shape: {parameter_values.shape}")
    
    # 2. Split data for regression
    print("\nStep 2: Creating train/validation split...")
    indices = np.arange(len(top_images))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    # Split data
    top_train, top_val = top_images[train_idx], top_images[val_idx]
    side_train, side_val = side_images[train_idx], side_images[val_idx]
    param_train, param_val = parameter_values[train_idx], parameter_values[val_idx]
    
    print(f"Training set: {len(top_train)} samples")
    print(f"Validation set: {len(top_val)} samples")
    
    # 3. Create regression model
    print("\nStep 3: Creating patent-based regression model...")
    model = create_patent_regression_model(input_shape=(224, 224, 3))
    
    # Display model summary
    print("Model Architecture Summary:")
    print(f"Total Parameters: {model.count_params():,}")
    print(f"Input: Dual-view images + continuous parameter prediction")
    print(f"Output: 4 continuous parameters [弯曲强度, 强度, 形变强度, 形变率]")
    
    # 4. Compile with parameter-weighted loss
    print("\nStep 4: Compiling model with parameter-weighted loss...")
    
    # Parameter importance weights from patent
    parameter_weights = np.array([1.0, 0.6, 0.6, 0.3], dtype=np.float32)
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=parameter_weighted_mse_loss(parameter_weights),
        metrics=[
            'mae',
            parameter_weighted_mae_loss(parameter_weights)
        ]
    )
    
    # 5. Setup callbacks
    print("\nStep 5: Setting up training callbacks...")
    callbacks = create_regression_callbacks()
    
    # 6. Train model
    print("\nStep 6: Starting regression training...")
    print("Training with:")
    print("- Dual-view image inputs (top + side)")
    print("- Parameter-weighted MSE loss")
    print("- Patent-based importance weighting")
    
    history = model.fit(
        [top_train, side_train],
        param_train,
        validation_data=([top_val, side_val], param_val),
        batch_size=16,
        epochs=100,
        callbacks=callbacks,
        verbose=1
    )
    
    # 7. Evaluate model
    print("\nStep 7: Evaluating regression model...")
    results = evaluate_regression_model(
        model, 
        [top_val, side_val], 
        param_val,
        param_scaler=data_loader.param_scaler
    )
    
    # 8. Fine-tuning phase
    print("\nStep 8: Fine-tuning with unfrozen CNN layers...")
    
    # Unfreeze ResNet50 layers for fine-tuning
    for layer in model.layers:
        if 'resnet50' in layer.name:
            layer.trainable = True
            # Freeze early layers, unfreeze later ones
            for sublayer in layer.layers[:-30]:
                sublayer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss=parameter_weighted_mse_loss(parameter_weights),
        metrics=['mae', parameter_weighted_mae_loss(parameter_weights)]
    )
    
    # Fine-tune
    fine_tune_history = model.fit(
        [top_train, side_train],
        param_train,
        validation_data=([top_val, side_val], param_val),
        batch_size=8,
        epochs=30,
        callbacks=callbacks,
        verbose=1
    )
    
    # 9. Final evaluation
    print("\nStep 9: Final evaluation after fine-tuning...")
    final_results = evaluate_regression_model(
        model,
        [top_val, side_val], 
        param_val,
        param_scaler=data_loader.param_scaler
    )
    
    # 10. Save final model
    model.save('patent_regression_model_final.h5')
    print("Model saved as 'patent_regression_model_final.h5'")
    
    # 11. Plot results
    print("\nStep 10: Plotting regression results...")
    plot_regression_results(final_results)
    
    # 12. Plot training history
    plot_training_history(history, fine_tune_history)
    
    print("\n" + "="*80)
    print("Regression training completed successfully!")
    print("Files created:")
    print("- best_patent_regression_model.h5 (best model during training)")
    print("- patent_regression_model_final.h5 (final fine-tuned model)")
    print("- regression_results.png (prediction scatter plots)")
    print("- regression_training_history.png (training curves)")
    print("="*80)
    
    return model, history, fine_tune_history, final_results

def plot_training_history(history, fine_tune_history=None):
    """Plot training history for regression model"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss', color='blue')
    ax1.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    
    if fine_tune_history:
        epochs_offset = len(history.history['loss'])
        ax1.plot(range(epochs_offset, epochs_offset + len(fine_tune_history.history['loss'])),
                fine_tune_history.history['loss'], label='Fine-tune Training', linestyle='--', color='blue')
        ax1.plot(range(epochs_offset, epochs_offset + len(fine_tune_history.history['val_loss'])),
                fine_tune_history.history['val_loss'], label='Fine-tune Validation', linestyle='--', color='orange')
    
    ax1.set_title('Model Loss (Parameter-Weighted MSE)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot MAE
    ax2.plot(history.history['mae'], label='Training MAE', color='green')
    ax2.plot(history.history['val_mae'], label='Validation MAE', color='red')
    
    if fine_tune_history:
        epochs_offset = len(history.history['mae'])
        ax2.plot(range(epochs_offset, epochs_offset + len(fine_tune_history.history['mae'])),
                fine_tune_history.history['mae'], label='Fine-tune Training MAE', linestyle='--', color='green')
        ax2.plot(range(epochs_offset, epochs_offset + len(fine_tune_history.history['val_mae'])),
                fine_tune_history.history['val_mae'], label='Fine-tune Validation MAE', linestyle='--', color='red')
    
    ax2.set_title('Model MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('regression_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Run regression training
    model, history, fine_tune_history, results = train_patent_regression_model()