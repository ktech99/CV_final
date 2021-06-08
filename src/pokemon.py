#importing turi create and naming it tc for easier use later
import turicreate as tc
#loading the images into turicreate

def train_kaggle():
    data_kaggle = tc.image_analysis.load_images(
    #the path for your image folder (change it to your own path)
    'study/Cse455_CV/final_project/PokemonData',
    with_path=True)

    data_kaggle['label'] = data_kaggle['path'].apply(
        #labelling the image as a cat if the folder name is cat, labelling it as a dog if it isn't
        lambda path: path[69: path.rindex("/")]
    )
    
    training_data, testing_data = data_kaggle.random_split(0.80)
    print("training with the kaggle dataset")
    #Training the model
    model_kaggle = tc.image_classifier.create(
    training_data,
    target='label',
    #using the squeezenet model for mobiles, for more accurate results use resnet-50
    model='squeezenet_v1.1',
    #number of times turicreate should attempt to train the model
    max_iterations=3000
    )

    #used to check the accuracy of our trained model using the images we had set aside
    metrics = model_kaggle.evaluate(testing_data)
    print("test accuracy of kaggle dataset")

    return metrics['accuracy']

def train_custom():
    data_custom = tc.image_analysis.load_images(
    #the path for your image folder (change it to your own path)
    'study/Cse455_CV/final_project/images',
    with_path=True)

    data_custom['label'] = data_custom['path'].apply(
    #labelling the image as a cat if the folder name is cat, labelling it as a dog if it isn't
    lambda path: path[63: path.rindex("/")] 
    )
    
    training_data, testing_data = data_custom.random_split(0.80)
    print("training with the kaggle dataset")
    #Training the model
    model_kaggle = tc.image_classifier.create(
    training_data,
    target='label',
    #using the squeezenet model for mobiles, for more accurate results use resnet-50
    model='squeezenet_v1.1',
    #number of times turicreate should attempt to train the model
    max_iterations=3000
    )

    #used to check the accuracy of our trained model using the images we had set aside
    metrics = model_kaggle.evaluate(testing_data)
    print("test accuracy of kaggle dataset")

    return metrics['accuracy']

def train_custom_kaggle():
    data_mixed = tc.image_analysis.load_images(
    #the path for your image folder (change it to your own path)
    'study/Cse455_CV/final_project/mixed',
    with_path=True
)

    data_mixed['label'] = data_mixed['path'].apply(
    #labelling the image as a cat if the folder name is cat, labelling it as a dog if it isn't
    lambda path: path[62: path.rindex("/")] 
    )
    
    training_data, testing_data = data_mixed.random_split(0.80)
    print("training with the kaggle dataset")
    #Training the model
    model_kaggle = tc.image_classifier.create(
    training_data,
    target='label',
    #using the squeezenet model for mobiles, for more accurate results use resnet-50
    model='squeezenet_v1.1',
    #number of times turicreate should attempt to train the model
    max_iterations=3000
    )

    #used to check the accuracy of our trained model using the images we had set aside
    metrics = model_kaggle.evaluate(testing_data)
    print("test accuracy of kaggle dataset")

    return metrics['accuracy']


def main():
    accuracy = train_custom_kaggle()
    print("\n\n\n\n\n\n\n")
    print("completed training custom + kaggle dataset")
    print("testing accuracy: %d", accuracy)


if __name__ == "__main__":
    main()