import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# The dataset is loaded from a CSV file using pandas' read_csv() function and stored in a DataFrame called "df".
df = pd.read_csv('MushroomCSV.csv')

# Split data into features and target
X = df.drop('CLASS', axis=1)
y = df['CLASS']

# Convert categorical features to numerical
X = pd.get_dummies(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Naïve Bayes model
nb = GaussianNB()
nb.fit(X_train, y_train)

# Predict the classes of the test set
y_pred = nb.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Define function to classify mushroom based on user input
def classify_mushroom():
    # Load dataset
    df = pd.read_csv('MushroomCSV.csv')

    # Get possible values for each feature
    features = ['CAPSHAPE', 'SURFACE', 'COLOR', 'BRUISES', 'ODOR',
                'GILL-ATTACHMENT', 'GILL-SPACING', 'GILL-SIZE', 'GILL-COLOR',
                'STALK-SHAPE', 'STALK-ROOT', 'STALK-SURFACE-ABOVE-RING', 'STALK-SURFACE-BELOW-RING',
                'STALK-COLOR ABOVE-RING', 'STALK-COLOR-BELOW-RING', 'VEIL-COLOR',
                'RING-NUMBER', 'RING-TYPE', 'SPORE-PRINT-COLOR', 'POPULATION', 'HABITAT']
    feature_values = {feature: df[feature].unique() for feature in features}

    # Get user input
    user_input = {}
    for feature in features:
        while True:
            value = input(f"Enter {feature.lower()} ({', '.join(feature_values[feature])}): ")
            if value in feature_values[feature]:
                user_input[feature] = value
                break
            else:
                print(f"Invalid value for {feature.lower()}, please try again.")

    # Convert user input to dataframe with numerical values
    user_input = pd.DataFrame(user_input, index=[0])
    user_input = pd.get_dummies(user_input)

    # Ensure the user input is encoded in the same way as the training data
    X_cols = X.columns
    user_input = user_input.reindex(columns=X_cols, fill_value=0)

    # Predict the class of the user input
    y_pred = nb.predict(user_input)

    # Print the predicted class
    if y_pred[0] == 'edible':
        print("Classification: edible")
    else:
        print("Classification: poisonous")
classify_mushroom()