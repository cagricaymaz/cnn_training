import numpy as np
from glob import glob
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.core import Dense, Flatten
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.applications.resnet import ResNet101, ResNet50

class PracticeObject:
    def __init__(self, name, image_shape, input_shape, model):
        self.name = name
        self.image_shape = image_shape
        self.input_shape = input_shape
        self.model = model


def customModel1(input_shape):
    model = Sequential([
        Conv2D(16, (3,3), activation='relu', input_shape = input_shape),
        MaxPooling2D(2,2),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(2,activation='softmax')
    ])
    return model

def customModel2(input_shape):

    model = Sequential([
        Conv2D(32, (3,3), activation = 'relu', input_shape = input_shape),
        Conv2D(32, (3,3), activation = 'relu'),
        Conv2D(32, (3,3), activation = 'relu'),
        MaxPooling2D(pool_size = (2,2)),
        Dropout(0.3),
        Conv2D(64, (3,3), activation ='relu'),
        Conv2D(64, (3,3), activation ='relu'),
        Conv2D(64, (3,3), activation ='relu'),
        MaxPooling2D(pool_size = (2,2)),
        Dropout(0.3),
        Conv2D(128, (3,3), activation ='relu'),
        Conv2D(128, (3,3), activation ='relu'),
        Conv2D(128, (3,3), activation ='relu'),
        MaxPooling2D(pool_size = (2,2)),
        Dropout(0.3),
        Flatten(),
        Dense(256, activation = "relu"),
        Dropout(0.3),
        Dense(2, activation = "softmax"),
    ]) 
    
    return model
    
def customModel3(input_shape):

    model = Sequential([
        Conv2D(32, (3,3), activation = 'relu', input_shape = input_shape),
        BatchNormalization(),
        Conv2D(32, (3,3), activation = 'relu'),
        BatchNormalization(),
        Conv2D(32, (3,3), activation = 'relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size = (2,2)),
        Dropout(0.3),
        Conv2D(64, (3,3), activation ='relu'),
        BatchNormalization(),
        Conv2D(64, (3,3), activation ='relu'),
        BatchNormalization(),
        Conv2D(64, (3,3), activation ='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size = (2,2)),
        Dropout(0.3),
        Conv2D(128, (3,3), activation ='relu'),
        BatchNormalization(),
        Conv2D(128, (3,3), activation ='relu'),
        BatchNormalization(),
        Conv2D(128, (3,3), activation ='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size = (2,2)),
        Dropout(0.3),
        Flatten(),
        Dense(256, activation = "relu"),
        Dropout(0.3),
        Dense(2, activation = "softmax"),
    ])
    
    return model

def main():

    practices = [
        PracticeObject(
            name = 'CustomModel1',
            image_shape = (50,50), 
            input_shape = (50, 50, 3),
            model = customModel1(input_shape=(50,50,3))),
        PracticeObject(
            name = 'CustomModel2',
            image_shape = (50,50), 
            input_shape = (50, 50, 3),
            model = customModel2(input_shape=(50,50,3))),
        PracticeObject(
            name = 'CustomModel3',
            image_shape = (50,50), 
            input_shape = (50, 50, 3),
            model = customModel3(input_shape=(50,50,3))),
        PracticeObject(
            name = 'ResNet50', 
            image_shape = (224,224), 
            input_shape = (224, 224, 3), 
            model = ResNet50(input_shape=(224, 224, 3), weights=None, classes=2)),
        PracticeObject(
            name = 'ResNet101', 
            image_shape = (224,224), 
            input_shape = (224, 224, 3), 
            model = ResNet101(input_shape=(224, 224, 3), weights=None, classes=2)),
        PracticeObject(
            name = 'VGG16', 
            image_shape = (50,50), 
            input_shape = (50, 50, 3), 
            model = VGG16(input_shape=(50, 50, 3), weights=None, classes=2)),   
        PracticeObject(
            name = 'VGG19', 
            image_shape = (50,50), 
            input_shape = (50, 50, 3), 
            model = VGG19(input_shape=(50, 50, 3), weights=None, classes=2)),  
         PracticeObject(
            name = 'InceptionV3', 
            image_shape = (75,75), 
            input_shape = (75, 75, 3), 
            model = InceptionV3(input_shape=(75, 75, 3), weights=None, classes=2)),  
         PracticeObject(
            name = 'InceptionResNetV2', 
            image_shape = (75,75), 
            input_shape = (75, 75, 3), 
            model = InceptionResNetV2(input_shape=(75, 75, 3), weights=None, classes=2)), 
    ]
    batch_sizes = [
        1,
        16,
        32
    ]
    optimizers = [
        'adam',
        'gradient_descent',
        'rmsprop',
        'adagrad',
        'nadam'
    ]
    losses = [
        'binary_crossentropy',
        'categorical_crossentropy',
        'sparse_categorical_crossentropy'
    ]
    classes = [
        'negative',
        'positive'
    ]

    image_generator = ImageDataGenerator(
        rescale = 1./255, 
        validation_split = 0.15)

    output_file = open('outputs.txt','w')

    for practice in practices:

        current_loss = None
        current_optimizer = None
        current_batch_size = None

        for batch_size in batch_sizes:
            for optimizer in optimizers:
                for loss in losses:
                    try:
                        current_loss = loss
                        current_optimizer = optimizer
                        current_batch_size = batch_size

                        train_data = image_generator.flow_from_directory(
                            'data/train_small',
                            target_size = practice.image_shape,
                            batch_size = current_batch_size,
                            class_mode ='categorical',
                            classes = classes,
                            subset = 'training',
                            shuffle=True)
                        val_data = image_generator.flow_from_directory(
                            'data/train_small',
                            target_size = practice.image_shape,
                            batch_size = current_batch_size,
                            class_mode = 'categorical',
                            classes = classes,
                            subset = 'validation',
                            shuffle=True)
                        test_data = image_generator.flow_from_directory(
                            'data/test_small',
                            target_size = practice.image_shape,
                            batch_size = current_batch_size,
                            class_mode = 'categorical',
                            classes = classes)

   

                        practice.model.compile(
                            optimizer = current_optimizer, 
                            loss = current_loss,
                            metrics = ['accuracy'])

                        history = practice.model.fit(
                            train_data,
                            steps_per_epoch = np.ceil(train_data.n / current_batch_size),
                            validation_data = val_data,
                            validation_steps = np.ceil(val_data.n / current_batch_size))

                        practice.model.save('model_data/{}'.format(practice.name))

                        output_file.writelines([
                            '[SUCCESS]\n',
                            'Model : {}\n'.format(practice.name),
                            '==========================================\n',
                            '    Loss       = {}\n'.format(current_loss),
                            '    Optimizer  = {}\n'.format(current_optimizer),
                            '    Batch Size = {}\n'.format(current_batch_size),
                            '    ----\n'
                            '    val_accuracy = {}\n'.format(history.history['val_accuracy']),
                            '==========================================\n\n'
                        ])

                    except:
    
                        output_file.writelines([
                            '[FAIL]\n',
                            'Model : {}\n'.format(practice.name),
                            '==========================================\n',
                            '    Loss       = {}\n'.format(current_loss),
                            '    Optimizer  = {}\n'.format(current_optimizer),
                            '    Batch Size = {}\n'.format(current_batch_size),
                            '==========================================\n\n'
                        ])
    
    return

if __name__ == "__main__":
    main()


