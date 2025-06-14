# model_train.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import numpy as np # For class weights if needed, and general utility

# --- Configuration ---
# Ensure these paths are correct and use raw strings (r'...') for Windows paths
DATASET_PATH = r'F:\Mohan files\Sixth sem\NM\India food dataset'
EXISTING_MODEL_PATH = r'F:\Mohan files\Sixth sem\NM\model1.h5'  # YOUR ORIGINAL 9-CLASS MODEL
NEW_MODEL_SAVE_PATH = r'F:\Mohan files\Sixth sem\NM\model1_finetuned.h5' # This file will be CREATED

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32  # Adjust based on your GPU memory
EPOCHS = 25      # Number of epochs for fine-tuning. Start with 15-25, observe, and adjust.
LEARNING_RATE = 0.00005 # Use a very small learning rate for fine-tuning (e.g., 1e-5 or 5e-5)

# --- 1. Data Augmentation and Loading ---
# We'll also create a validation split from your training data
# More aggressive augmentation can be useful if your dataset is small
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,      # Increased range
    width_shift_range=0.25, # Increased range
    height_shift_range=0.25,# Increased range
    shear_range=0.2,
    zoom_range=0.25,        # Increased range
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2], # Added brightness augmentation
    validation_split=0.2  # Reserve 20% of data for validation
)

# Separate generator for validation data - only rescaling, no augmentation
validation_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2 # Must be same as in train_datagen for proper split
)

print("Setting up training data generator...")
try:
    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',  # Specify this is the training data
        shuffle=True
    )
except FileNotFoundError:
    print(f"ERROR: Dataset directory not found at {DATASET_PATH}")
    print("Please ensure the path is correct and the directory exists.")
    exit()

print("Setting up validation data generator...")
try:
    validation_generator = validation_datagen.flow_from_directory( # Use validation_datagen here
        DATASET_PATH,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',  # Specify this is the validation data
        shuffle=False # No need to shuffle validation data
    )
except FileNotFoundError:
    # This error should have been caught by the train_generator setup, but good to have
    print(f"ERROR: Dataset directory not found at {DATASET_PATH} for validation.")
    exit()


if train_generator.samples == 0:
    print(f"ERROR: No training images found in {DATASET_PATH} with the specified subset configuration.")
    print("Please check your dataset structure and that it contains images in subdirectories.")
    exit()
if validation_generator.samples == 0:
    print(f"ERROR: No validation images found. This might happen if your dataset is too small for a 20% split,")
    print(f"or if 'validation_split' is not correctly configured in both ImageDataGenerators.")
    exit()


print(f"Found {train_generator.samples} images for training from {len(train_generator.class_indices)} classes.")
print(f"Found {validation_generator.samples} images for validation from {len(validation_generator.class_indices)} classes.")
print(f"Training Class indices: {train_generator.class_indices}")
print(f"Validation Class indices: {validation_generator.class_indices}") # Should be the same as training

# --- 2. Load Existing Model for Fine-Tuning ---
print(f"\nLoading existing model from: {EXISTING_MODEL_PATH}")
if not os.path.exists(EXISTING_MODEL_PATH):
    print(f"ERROR: Existing model file not found at {EXISTING_MODEL_PATH}")
    print("This script requires model1.h5 to exist for fine-tuning.")
    exit()

try:
    model = load_model(EXISTING_MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure the model path is correct and the model file is not corrupted.")
    print("If model1.h5 was not a Keras model or uses custom objects not defined, loading will fail.")
    exit()

# --- 3. Verify Model and Dataset Compatibility ---
num_classes_dataset = len(train_generator.class_indices)
model_output_units = model.output_shape[-1]

if model_output_units != num_classes_dataset:
    print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"ERROR: CRITICAL MISMATCH!")
    print(f"The loaded model ('{os.path.basename(EXISTING_MODEL_PATH)}') has an output layer with {model_output_units} units (for {model_output_units} classes).")
    print(f"However, your dataset in '{DATASET_PATH}' was found to have {num_classes_dataset} classes.")
    print(f"These numbers MUST match for fine-tuning.")
    print(f"Dataset classes found: {list(train_generator.class_indices.keys())}")
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("\nPossible Solutions:")
    print(f"1. Ensure your '{DATASET_PATH}' contains subdirectories for ALL {model_output_units} classes that '{os.path.basename(EXISTING_MODEL_PATH)}' was originally trained on.")
    print(f"2. If you intend to train for a different number of classes ({num_classes_dataset}), you must REBUILD the final classification layer of the model (more advanced).")
    exit()
else:
    print(f"\nModel and dataset class count match: {num_classes_dataset} classes. Proceeding with fine-tuning.")


# Optional: Make only some layers trainable (fine-tuning strategy)
# For example, if model1.h5 is a large pre-trained network like VGG16 or ResNet
# you might freeze the base and only train the top layers you added, or unfreeze a few more.
# If model1.h5 is your own custom CNN, you might fine-tune all layers or unfreeze later ones.

# Example: Unfreeze all layers for full fine-tuning (often suitable for custom models)
for layer in model.layers:
    layer.trainable = True
print("\nAll layers in the model have been set to trainable for fine-tuning.")

# Example: Unfreeze only the last N layers
# NUM_LAYERS_TO_UNFREEZE = 5
# for layer in model.layers[:-NUM_LAYERS_TO_UNFREEZE]:
#    layer.trainable = False
# for layer in model.layers[-NUM_LAYERS_TO_UNFREEZE:]:
#    layer.trainable = True
# print(f"\nLast {NUM_LAYERS_TO_UNFREEZE} layers set to trainable. Base layers frozen.")


# --- 4. Compile the Model for Fine-Tuning ---
# It's crucial to recompile the model after changing layer trainability or loading a model.
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), # Use a small learning rate
              loss='categorical_crossentropy',
              metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]) # Added top-3 accuracy

print("\nModel summary (after setting trainability):")
model.summary()

# --- 5. Train (Fine-Tune) the Model ---
print(f"\nStarting fine-tuning for {EPOCHS} epochs...")
print(f"Batch size: {BATCH_SIZE}, Learning rate: {LEARNING_RATE}")
print(f"Training samples: {train_generator.samples}, Validation samples: {validation_generator.samples}")

# Optional: Add Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_lr=1e-7),
    tf.keras.callbacks.ModelCheckpoint(filepath=NEW_MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1) # Saves the best model
]

history = model.fit(
    train_generator,
    steps_per_epoch=max(1, train_generator.samples // BATCH_SIZE), # Ensure at least 1 step
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=max(1, validation_generator.samples // BATCH_SIZE), # Ensure at least 1 step
    callbacks=callbacks
)

# The best model during training (based on val_accuracy) would have been saved by ModelCheckpoint.
# If you want to explicitly save the *final* state (even if not the best), you can do:
# model.save(NEW_MODEL_SAVE_PATH.replace('.h5', '_final_epoch.h5'))
print(f"\nFine-tuning complete. Best model saved to: {NEW_MODEL_SAVE_PATH} (if val_accuracy improved).")


# --- 6. Evaluate the Model (Optional, on validation set) ---
print("\nEvaluating the fine-tuned model on the validation set (using weights from best epoch if EarlyStopping restored them or ModelCheckpoint saved):")
# If ModelCheckpoint saved the best model, you might want to load it explicitly before evaluation
# if 'ModelCheckpoint' in str(callbacks): # crude check
#    print(f"Loading best model from {NEW_MODEL_SAVE_PATH} for final evaluation.")
#    model = load_model(NEW_MODEL_SAVE_PATH) # Load the best saved model

val_loss, val_accuracy, val_top_3_accuracy = model.evaluate(validation_generator, steps=max(1, validation_generator.samples // BATCH_SIZE))
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy*100:.2f}%")
print(f"Validation Top-3 Accuracy: {val_top_3_accuracy*100:.2f}%")


# --- 7. Plot Training History ---
if history and history.history:
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('validation_accuracy', []) # Keras 3 uses 'validation_accuracy'
    if not val_acc: # Fallback for older Keras/TF
        val_acc = history.history.get('val_accuracy', [])

    loss = history.history.get('loss', [])
    val_loss = history.history.get('validation_loss', []) # Keras 3 uses 'validation_loss'
    if not val_loss: # Fallback for older Keras/TF
        val_loss = history.history.get('val_loss', [])

    epochs_ran = range(len(acc))

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    if acc and val_acc:
        plt.plot(epochs_ran, acc, label='Training Accuracy')
        plt.plot(epochs_ran, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
    else:
        plt.text(0.5, 0.5, 'Accuracy data not available', ha='center', va='center')


    plt.subplot(1, 2, 2)
    if loss and val_loss:
        plt.plot(epochs_ran, loss, label='Training Loss')
        plt.plot(epochs_ran, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
    else:
        plt.text(0.5, 0.5, 'Loss data not available', ha='center', va='center')

    plt.suptitle(f"Fine-tuning Results: {os.path.basename(NEW_MODEL_SAVE_PATH)}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
    
    plot_save_path = NEW_MODEL_SAVE_PATH.replace('.h5', '_training_history.png')
    try:
        plt.savefig(plot_save_path)
        print(f"Training history plot saved to: {plot_save_path}")
    except Exception as e:
        print(f"Could not save training plot: {e}")
    plt.show()
else:
    print("No training history to plot.")

print("\nScript finished.")
print(f"If training was successful and the model improved, update your 'ai_service.py'")
print(f"to use the new model: '{NEW_MODEL_SAVE_PATH}'")