from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    r'F:\Mohan files\Sixth sem\NM\India food dataset',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training')

val_generator = train_datagen.flow_from_directory(
    r'F:\Mohan files\Sixth sem\NM\India food dataset',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation')

