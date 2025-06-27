from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from preprocess import preprocess_data  # Ensure this import is correct

# Directory paths
data_dir = "data/train"  # Path to your training data
model_save_path = "models/model.h5"  # Path to save the trained model

# Preprocess the data
train_generator, validation_generator = preprocess_data(
    src_dir=data_dir,
    img_size=(224, 224),
    batch_size=32,
    validation_split=0.2
)

# Define the CNN model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')  # Multi-class classification
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator
)

# Save the trained model
model.save(model_save_path)
print(f"Model saved to {model_save_path}")
