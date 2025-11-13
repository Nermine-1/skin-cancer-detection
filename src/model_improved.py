from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

def build_model(input_shape=(128, 128, 3), num_classes=7, learning_rate=0.001):
    """
    Build an improved CNN model optimized for larger images (128x128).
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes
        learning_rate: Initial learning rate
    
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3,3), padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2,2),
        Dropout(0.2),
        
        # Second convolutional block
        Conv2D(64, (3,3), padding='same'),
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2,2),
        Dropout(0.2),
        
        # Third convolutional block
        Conv2D(128, (3,3), padding='same'),
        BatchNormalization(),
        Conv2D(128, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2,2),
        Dropout(0.3),
        
        # Fourth convolutional block
        Conv2D(256, (3,3), padding='same'),
        BatchNormalization(),
        Conv2D(256, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2,2),
        Dropout(0.3),
        
        # Use GlobalAveragePooling instead of Flatten for better generalization
        GlobalAveragePooling2D(),
        
        # Fully connected layers
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

