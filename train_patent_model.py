"""
Training Script for Patent-Based Material Recognition Model
Implements the complete training pipeline following patent methodology
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from patent_based_model import (PatentDataLoader, create_patent_multimodal_model, 
                               physics_informed_loss, create_data_augmentation)

def parameter_aware_split(parameters, labels, test_size=0.2, random_state=42):
    """
    Split data while maintaining parameter diversity across train/test sets
    Ensures all parameter ranges are represented in both sets
    """
    
    # Create parameter-based stratification groups
    param_groups = []
    for params in parameters:
        # Group by rounded parameter values to ensure diversity
        group = tuple(np.round(np.array(params) / 10) * 10)  # Round to nearest 10
        param_groups.append(group)
    
    # Use stratified split based on both labels and parameter groups
    combined_stratify = [f"{label}_{group}" for label, group in zip(labels, param_groups)]
    
    try:
        train_idx, test_idx = train_test_split(
            range(len(labels)), 
            test_size=test_size, 
            random_state=random_state,
            stratify=combined_stratify
        )
    except ValueError:
        # Fallback to simple stratification by labels only if combined fails
        print("Using simple label-based stratification (some parameter groups too small)")
        train_idx, test_idx = train_test_split(
            range(len(labels)), 
            test_size=test_size, 
            random_state=random_state,
            stratify=labels
        )
    
    return train_idx, test_idx

def plot_parameter_distribution(parameters, labels, title="Parameter Distribution"):
    """Visualize parameter distribution across classes"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    param_names = ['弯曲强度 (Bending Strength)', '强度 (Strength)', 
                   '形变强度 (Deformation Strength)', '形变率 (Deformation Rate)']
    
    for i, (ax, param_name) in enumerate(zip(axes.flat, param_names)):
        param_values = parameters[:, i]
        ax.hist(param_values, bins=20, alpha=0.7, edgecolor='black')
        ax.set_xlabel(param_name)
        ax.set_ylabel('Frequency')
        ax.set_title(f'{param_name} Distribution')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('parameter_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def train_patent_model():
    """Main training function following patent methodology"""
    
    print("="*60)
    print("Patent-Based Material Recognition Training")
    print("Based on: 一种基于AI图形识别技术的数字面料物理属性获取方法")
    print("="*60)
    
    # 1. Load patent-compliant data
    print("Step 1: Loading dual-view data with physical parameters...")
    data_loader = PatentDataLoader('./Materials_data', image_size=(224, 224))
    
    top_images, side_images, parameters, labels, label_names = data_loader.load_dual_view_data()
    
    if len(top_images) == 0:
        print("ERROR: No valid image pairs found!")
        print("Please ensure your data structure follows the patent format:")
        print("- Folders named like: '1-20.20.20.20-1'")
        print("- Images named like: 'filename-1.jpg' (top) and 'filename-2.jpg' (side)")
        return None
    
    print(f"✓ Loaded {len(top_images)} image pairs")
    print(f"✓ Found {len(np.unique(labels))} material classes")
    print(f"✓ Parameter shape: {parameters.shape}")
    
    # Visualize parameter distribution
    plot_parameter_distribution(parameters, labels)
    
    # 2. Encode labels
    print("\nStep 2: Encoding labels...")
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    
    print(f"✓ {num_classes} classes encoded")
    print("Class distribution:", Counter(encoded_labels))
    
    # 3. Parameter-aware data splitting
    print("\nStep 3: Creating parameter-aware train/test split...")
    train_idx, val_idx = parameter_aware_split(parameters, encoded_labels, test_size=0.2)
    
    # Split data
    top_train, top_val = top_images[train_idx], top_images[val_idx]
    side_train, side_val = side_images[train_idx], side_images[val_idx]
    param_train, param_val = parameters[train_idx], parameters[val_idx]
    label_train, label_val = encoded_labels[train_idx], encoded_labels[val_idx]
    
    print(f"✓ Training set: {len(top_train)} samples")
    print(f"✓ Validation set: {len(top_val)} samples")
    
    # 4. Compute parameter similarity matrix
    print("\nStep 4: Computing parameter similarity matrix...")
    param_similarity = data_loader.compute_parameter_similarity(param_train)
    print(f"✓ Similarity matrix shape: {param_similarity.shape}")
    
    # 5. Create model
    print("\nStep 5: Creating patent-based multi-modal model...")
    model = create_patent_multimodal_model(num_classes, input_shape=(224, 224, 3))
    
    # Display model architecture
    print("Model Architecture:")
    model.summary()
    
    # 6. Compile model with physics-informed loss
    print("\nStep 6: Compiling model with physics-informed loss...")
    
    # Compute class weights for imbalanced dataset
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(label_train), 
        y=label_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    # Custom loss with parameter similarity
    custom_loss = physics_informed_loss(
        tf.constant(param_similarity, dtype=tf.float32), 
        alpha=0.15
    )
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',  # Using standard loss for stability
        metrics=['accuracy']
    )
    
    # 7. Setup callbacks
    print("\nStep 7: Setting up training callbacks...")
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=20,
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
            'best_patent_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # 8. Data augmentation
    print("\nStep 8: Preparing data augmentation...")
    train_datagen = create_data_augmentation()
    
    # 9. Train model
    print("\nStep 9: Starting training...")
    print("Training with patent-based multi-modal architecture:")
    print("- Dual-view images (top + side)")
    print("- Physical parameters with importance weighting")
    print("- Parameter similarity consideration")
    
    history = model.fit(
        [top_train, side_train, param_train],
        label_train,
        validation_data=([top_val, side_val, param_val], label_val),
        batch_size=16,  # Smaller batch size for stability with multi-modal inputs
        epochs=100,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # 10. Evaluate model
    print("\nStep 10: Evaluating model performance...")
    val_predictions = model.predict([top_val, side_val, param_val])
    val_pred_classes = np.argmax(val_predictions, axis=1)
    
    print("Validation Classification Report:")
    print(classification_report(
        label_val, 
        val_pred_classes, 
        target_names=label_encoder.classes_
    ))
    
    # 11. Fine-tuning phase
    print("\nStep 11: Fine-tuning with unfrozen CNN layers...")
    
    # Unfreeze some layers for fine-tuning
    base_cnn = model.get_layer('resnet50')
    base_cnn.trainable = True
    
    # Freeze early layers, unfreeze later layers
    for layer in base_cnn.layers[:-20]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Fine-tune
    fine_tune_history = model.fit(
        [top_train, side_train, param_train],
        label_train,
        validation_data=([top_val, side_val, param_val], label_val),
        batch_size=8,  # Even smaller batch for fine-tuning
        epochs=20,
        callbacks=callbacks,
        verbose=1
    )
    
    # 12. Final evaluation
    print("\nStep 12: Final evaluation...")
    final_predictions = model.predict([top_val, side_val, param_val])
    final_pred_classes = np.argmax(final_predictions, axis=1)
    
    final_accuracy = np.mean(final_pred_classes == label_val)
    print(f"Final Validation Accuracy: {final_accuracy:.4f}")
    
    # Save final model
    model.save('patent_based_material_model.h5')
    print("✓ Model saved as 'patent_based_material_model.h5'")
    
    # 13. Plot training history
    plot_training_history(history, fine_tune_history)
    
    return model, history, fine_tune_history, label_encoder

def plot_training_history(history, fine_tune_history=None):
    """Plot training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    
    if fine_tune_history:
        epochs_offset = len(history.history['accuracy'])
        ax1.plot(range(epochs_offset, epochs_offset + len(fine_tune_history.history['accuracy'])),
                fine_tune_history.history['accuracy'], label='Fine-tune Training Accuracy', linestyle='--')
        ax1.plot(range(epochs_offset, epochs_offset + len(fine_tune_history.history['val_accuracy'])),
                fine_tune_history.history['val_accuracy'], label='Fine-tune Validation Accuracy', linestyle='--')
    
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    
    if fine_tune_history:
        epochs_offset = len(history.history['loss'])
        ax2.plot(range(epochs_offset, epochs_offset + len(fine_tune_history.history['loss'])),
                fine_tune_history.history['loss'], label='Fine-tune Training Loss', linestyle='--')
        ax2.plot(range(epochs_offset, epochs_offset + len(fine_tune_history.history['val_loss'])),
                fine_tune_history.history['val_loss'], label='Fine-tune Validation Loss', linestyle='--')
    
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('patent_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Run training
    model, history, fine_tune_history, label_encoder = train_patent_model()
    
    if model is not None:
        print("\n" + "="*60)
        print("Training completed successfully!")
        print("Files created:")
        print("- best_patent_model.h5 (best model during training)")
        print("- patent_based_material_model.h5 (final model)")
        print("- patent_training_history.png (training plots)")
        print("- parameter_distribution.png (data analysis)")
        print("="*60)