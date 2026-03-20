"""
Convert .keras models (Keras 3) to SavedModel format (compatible with all TF versions).
Run this on the SERVER where models were trained.

Usage:
    python convert_model.py
"""
import os
import sys
import zipfile
import tempfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Bidirectional, Conv1D, MaxPooling1D,
    GlobalAveragePooling1D, Dropout, BatchNormalization, Concatenate,
    Softmax, Multiply, Lambda, Activation, Add
)
from tensorflow.keras.models import Model

NUM_CLASSES = 5
INPUT_SHAPE = (126, 139)


# ============================================================================
# MODEL ARCHITECTURE (same as train.py)
# ============================================================================
def _gender_branch():
    inp = Input(shape=(1,), name='gender_input')
    g = Dense(16, activation='relu')(inp)
    g = BatchNormalization()(g)
    g = Dense(8, activation='relu')(g)
    return inp, g


def _attention_block(x):
    score = Dense(1)(x)
    weights = Softmax(axis=1)(score)
    context = Multiply()([x, weights])
    context = Lambda(lambda t: tf.reduce_sum(t, axis=1))(context)
    return context


def _classification_head(audio_feat, gender_feat, num_classes):
    concat = Concatenate()([audio_feat, gender_feat])
    reg = tf.keras.regularizers.l2(1e-4)
    z = Dense(256, activation='relu', kernel_regularizer=reg)(concat)
    z = BatchNormalization()(z)
    z = Dropout(0.4)(z)
    z = Dense(128, activation='relu', kernel_regularizer=reg)(z)
    z = BatchNormalization()(z)
    z = Dropout(0.4)(z)
    return Dense(num_classes, activation='softmax', name='output')(z)


def _residual_block(x, filters, kernel_size=3):
    shortcut = x
    out = Conv1D(filters, kernel_size, padding='same')(x)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv1D(filters, kernel_size, padding='same')(out)
    out = BatchNormalization()(out)
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    out = Add()([out, shortcut])
    out = Activation('relu')(out)
    return out


def build_bilstm_attention(input_shape, num_classes):
    audio_input = Input(shape=input_shape, name='audio_input')
    x = Bidirectional(LSTM(128, return_sequences=True))(audio_input)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = _attention_block(x)
    audio_feat = Dense(64, activation='relu')(x)
    audio_feat = BatchNormalization()(audio_feat)
    audio_feat = Dropout(0.4)(audio_feat)
    gender_input, gender_feat = _gender_branch()
    output = _classification_head(audio_feat, gender_feat, num_classes)
    return Model(inputs=[audio_input, gender_input], outputs=output)


def build_lstm_attention(input_shape, num_classes):
    audio_input = Input(shape=input_shape, name='audio_input')
    x = LSTM(128, return_sequences=True)(audio_input)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = LSTM(64, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = _attention_block(x)
    audio_feat = Dense(64, activation='relu')(x)
    audio_feat = BatchNormalization()(audio_feat)
    audio_feat = Dropout(0.4)(audio_feat)
    gender_input, gender_feat = _gender_branch()
    output = _classification_head(audio_feat, gender_feat, num_classes)
    return Model(inputs=[audio_input, gender_input], outputs=output)


def build_cnn_residual(input_shape, num_classes):
    audio_input = Input(shape=input_shape, name='audio_input')
    x = Conv1D(64, 5, padding='same', activation='relu')(audio_input)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)
    x = _residual_block(x, 64)
    x = Dropout(0.35)(x)
    x = _residual_block(x, 128)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.4)(x)
    x = _residual_block(x, 128)
    x = Dropout(0.4)(x)
    x = _residual_block(x, 128)
    x = Dropout(0.4)(x)
    x = GlobalAveragePooling1D()(x)
    audio_feat = Dense(64, activation='relu')(x)
    audio_feat = BatchNormalization()(audio_feat)
    audio_feat = Dropout(0.4)(audio_feat)
    gender_input, gender_feat = _gender_branch()
    output = _classification_head(audio_feat, gender_feat, num_classes)
    return Model(inputs=[audio_input, gender_input], outputs=output)


def build_crnn_attention(input_shape, num_classes):
    audio_input = Input(shape=input_shape, name='audio_input')
    x = Conv1D(64, 5, padding='same', activation='relu')(audio_input)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)
    x = _residual_block(x, 128)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.4)(x)
    x = _residual_block(x, 128)
    x = Dropout(0.4)(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = _attention_block(x)
    audio_feat = Dense(64, activation='relu')(x)
    audio_feat = BatchNormalization()(audio_feat)
    audio_feat = Dropout(0.4)(audio_feat)
    gender_input, gender_feat = _gender_branch()
    output = _classification_head(audio_feat, gender_feat, num_classes)
    return Model(inputs=[audio_input, gender_input], outputs=output)


# ============================================================================
# CONVERT
# ============================================================================
MODELS = {
    'BiLSTM_Attention': build_bilstm_attention,
    'LSTM_Attention': build_lstm_attention,
    'CNN_Residual': build_cnn_residual,
    'CRNN_Attention': build_crnn_attention,
}

def convert_model(keras_path, output_dir):
    """Rebuild model from code, load Keras3 weights, save as SavedModel."""
    basename = os.path.basename(keras_path)
    
    # Detect type
    model_type = None
    for key in MODELS:
        if key in basename:
            model_type = key
            break
    
    if model_type is None:
        print(f"  ❌ Cannot detect model type from: {basename}")
        return
    
    print(f"\n  Converting: {basename} (type: {model_type})")
    
    # Build fresh model
    builder = MODELS[model_type]
    model = builder(INPUT_SHAPE, NUM_CLASSES)
    
    # Load weights from .keras zip
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(keras_path, 'r') as z:
            z.extract('model.weights.h5', tmpdir)
            weights_path = os.path.join(tmpdir, 'model.weights.h5')
            model.load_weights(weights_path)
    
    print(f"  ✅ Weights loaded ({model.count_params():,} params)")
    
    # Save as SavedModel format
    save_name = basename.replace('.keras', '_saved')
    save_path = os.path.join(output_dir, save_name)
    model.export(save_path)
    print(f"  ✅ Saved to: {save_path}")
    
    return save_path


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output_optimized')
    
    print("=" * 50)
    print("  Model Converter: .keras → SavedModel")
    print("=" * 50)
    
    # Find all .keras files
    keras_files = [
        os.path.join(output_dir, f) 
        for f in os.listdir(output_dir) 
        if f.endswith('.keras')
    ]
    
    if not keras_files:
        print("  ❌ No .keras files found in output_optimized/")
        sys.exit(1)
    
    print(f"  Found {len(keras_files)} model(s)")
    
    for kf in keras_files:
        try:
            convert_model(kf, output_dir)
        except Exception as e:
            print(f"  ❌ Failed to convert {os.path.basename(kf)}: {e}")
    
    print(f"\n{'='*50}")
    print("  Done! Copy the *_saved/ folders to your local machine.")
    print(f"{'='*50}")
