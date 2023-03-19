# Mushroom Classification using Naïve Bayes
This code uses a Naïve Bayes classifier to classify mushrooms as either edible or poisonous based on various features. The code uses the pandas library to load the dataset from a CSV file and convert categorical features to numerical values. It then splits the data into training and testing sets, trains the Naïve Bayes model on the training data, and evaluates the model's accuracy on the testing data.

The code also defines a function `classify_mushroom()` that prompts the user to input various features of a mushroom and then predicts whether the mushroom is edible or poisonous using the trained model.

## Dependencies
This code requires the following libraries:

+ pandas
+ sklearn

## Dataset
The dataset used in this code is stored in the file MushroomCSV.csv. The dataset contains various features of mushrooms, such as cap shape, odor, and gill color, as well as a target variable indicating whether the mushroom is edible or poisonous.

## Usage
To use this code, simply run the script in a Python environment that has the required dependencies installed. The script will print the accuracy of the model on the testing data and then prompt the user to input various features of a mushroom to classify. The user input will be used to predict whether the mushroom is edible or poisonous.

Note: The script assumes that the dataset is stored in the file `MushroomCSV.csv` in the same directory as the script. If the dataset is stored elsewhere, the file path in the `pd.read_csv()` functions will need to be updated accordingly.
