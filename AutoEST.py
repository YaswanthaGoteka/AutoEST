from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load and preprocess data
data = 'car_data.csv'
df = pd.read_csv(data)

df['Car_Make'] = df['Car Make'].str.lower().str.strip()
df['Car_Model'] = df['Car Model'].str.upper().str.strip()
df['Condition_clean'] = df['Condition'].str.lower().str.strip()
df['Accident_clean'] = df['Accident'].str.upper().str.strip()
df['Color_clean'] = df['Color'].str.upper().str.strip()

# Label Encoding
le_make = LabelEncoder()
le_model = LabelEncoder()
le_year = LabelEncoder()
le_color = LabelEncoder()
le_mileage = LabelEncoder()
le_condition = LabelEncoder()
le_accident = LabelEncoder()

df['Make Encoded'] = le_make.fit_transform(df['Car_Make'])
df['Model Encoded'] = le_model.fit_transform(df['Car_Model'])
df['Color Encoded'] = le_color.fit_transform(df['Color_clean'])
df['Condition Encoded'] = le_condition.fit_transform(df['Condition_clean'])
df['Accident Encoded'] = le_accident.fit_transform(df['Accident_clean'])

# Features and target
features = ['Make Encoded', 'Model Encoded', 'Year', 'Color Encoded', 'Mileage', 'Condition Encoded',
            'Accident Encoded']
target = ['Price']
X = df[features]
y = df[target]

# Train random-forest-regressor model
rf = RandomForestRegressor(n_estimators=100, random_state=2025)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf.fit(X_train, y_train.values.ravel())
rf_prediction = rf.predict(X_test)

# Evaluations for model
mae = round(mean_absolute_error(y_test, rf_prediction), 2)
mse = round(mean_squared_error(y_test, rf_prediction), 2)
r2 = round(r2_score(y_test, rf_prediction), 2)
accuracy_percent = round((r2 * 100), 2)

print(f"MSE: {mse}")
print(f"MAE: {mae}")
print(f"R2 score: {r2}")
print(f"Accuracy: {accuracy_percent}%")


# Input section for new predictions
# Input section for new predictions
def get_user_input():
    print("\nEnter car details for price prediction:")

    # Get car make
    print(f"Available makes: {list(le_make.classes_)}")
    while True:
        car_make = input("Enter car make: ").lower().strip()
        if car_make in le_make.classes_:
            make_encoded = le_make.transform([car_make])[0]
            break
        print("Invalid make. Please choose from the available makes.")

    # Get car model
    print(f"Available models: {list(le_model.classes_)}")
    while True:
        car_model = input("Enter car model: ").upper().strip()
        if car_model in le_model.classes_:
            model_encoded = le_model.transform([car_model])[0]
            break
        print("Invalid model. Please choose from the available models.")

    # Show available years for the chosen model
    available_years = sorted(df[df['Car_Model'] == car_model]['Year'].unique())
    if len(available_years) > 0:
        print(f"Available years for {car_model}: {available_years}")
    else:
        print(f"⚠No specific years found in dataset for {car_model}, entering manually.")

    # Get year
    while True:
        try:
            year = int(input("Enter car year: "))
            if (len(available_years) == 0 and 1886 <= year <= 2025) or (year in available_years):
                break
            if len(available_years) > 0:
                print(f"Please choose a valid year from {available_years}.")
            else:
                print("Please enter a valid year between 1886 and 2025.")
        except ValueError:
            print("Please enter a numeric year.")

    # Get color
    print(f"Available colors: {list(le_color.classes_)}")
    while True:
        color = input("Enter car color: ").upper().strip()
        if color in le_color.classes_:
            color_encoded = le_color.transform([color])[0]
            break
        print("Invalid color. Please choose from the available colors.")

    # Get mileage
    while True:
        try:
            mileage = float(input("Enter car mileage (minimum 20000 for higher accuracy outputs): "))
            if mileage >= 0:
                break
            print("Mileage cannot be negative.")
        except ValueError:
            print("Please enter a valid number for mileage.")

    # Get condition
    print(f"Available conditions: {list(le_condition.classes_)}")
    while True:
        condition = input("Enter car condition: ").lower().strip()
        if condition in le_condition.classes_:
            condition_encoded = le_condition.transform([condition])[0]
            break
        print("Invalid condition. Please choose from the available conditions.")

    # Get accident history
    print(f"Available accident statuses: {list(le_accident.classes_)}")
    while True:
        accident = input("Enter accident history: ").upper().strip()
        if accident in le_accident.classes_:
            accident_encoded = le_accident.transform([accident])[0]
            break
        print("Invalid accident status. Please choose from the available statuses.")

    # Create input array
    input_data = np.array(
        [[make_encoded, model_encoded, year, color_encoded, mileage, condition_encoded, accident_encoded]])
    return input_data



# Predict price for user input
# Predict price for user input
while True:
    user_input = get_user_input()
    predicted_price = rf.predict(user_input)[0]

    # Apply different multipliers based on year
    car_year = user_input[0][2]  # year is the 3rd element in input_data
    current_year = 2025
    age = current_year - car_year

    if age <= 3:  # almost new
        lower_multiplier = 1.15
        upper_multiplier = 1.00
    elif age <= 7:  # moderately used
        lower_multiplier = 0.60
        upper_multiplier = 0.75
    elif age <= 12:  # older car
        lower_multiplier = 0.45
        upper_multiplier = 0.60
    else:  # very old
        lower_multiplier = 0.30
        upper_multiplier = 0.45

    predicted_range = f"${round(predicted_price * lower_multiplier, 2)} - ${round(predicted_price * upper_multiplier, 2)}"
    print(f"\nPredicted car price: {predicted_range}")

    continue_pred = input("\nWould you like to predict another car price? (yes/no): ").lower().strip()
    if continue_pred != 'yes':
        break
