from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_data(src_dir, img_size=(224, 224), batch_size=32, validation_split=0.2):
    """
    Preprocess the data and prepare it for training by creating image generators.
    """
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,           # Normalize pixel values to [0, 1]
        validation_split=validation_split,  # Reserve part of the dataset for validation
        rotation_range=30,          # Randomly rotate images by up to 30 degrees
        width_shift_range=0.2,      # Randomly shift images horizontally
        height_shift_range=0.2,     # Randomly shift images vertically
        shear_range=0.2,            # Randomly apply shearing
        zoom_range=0.2,             # Randomly zoom in on images
        horizontal_flip=True        # Randomly flip images horizontally
    )

    # Training data generator
    train_generator = datagen.flow_from_directory(
        src_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',  # Change to 'binary' if only two classes
        subset='training'          # Use the 'training' subset
    )

    # Validation data generator
    validation_generator = datagen.flow_from_directory(
        src_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',  # Change to 'binary' if only two classes
        subset='validation'        # Use the 'validation' subset
    )

    return train_generator, validation_generator
