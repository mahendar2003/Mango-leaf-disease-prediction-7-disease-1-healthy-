import tensorflow as tf
from preprocess import preprocess_data

# Load the trained model
model_path = "models/model.h5"
model = tf.keras.models.load_model(model_path)

# Directory for evaluation
data_dir = "data/train"  # Use the same directory for validation

# Preprocess the validation data
_, validation_generator = preprocess_data(
    src_dir=data_dir,
    img_size=(224, 224),
    batch_size=32,
    validation_split=0.2
)

# Evaluate the model
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
print(f"Validation Loss: {loss:.4f}")
