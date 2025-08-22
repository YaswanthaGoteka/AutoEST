# AutoEST a Car Price Prediction Model using Random Forest Regressor

This project is a **machine learning application** that predicts the price of a car based on user-provided details such as make, model, year, color, mileage, condition, and accident history.  
It uses a **Random Forest Regressor** trained on car data that I aquired from Kaggle Datasets.

Specifically: Hamidreza Naderbeygi: Cars for Sale from Kaggle
---

## Features
Predicts car prices based on multiple attributes:
  - Car Make (e.g., Audi, BMW, Toyota)
  - Car Model (e.g., A3, X5, Corolla)
  - Year of Manufacture (validated against available dataset years)
  - Color
  - Mileage
  - Condition (`new`, `like new`, `used`)
  - Accident history (`YES`, `NO`)
Provides model evaluation metrics:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - R² Score
  - Accuracy percentage (based on R² score)
Interactive **user input system** for making predictions.
Ensures input validation (e.g., valid years, models, conditions).

---

## 🛠️ Technologies Used
- **Python in PyCharm**
- **Pandas** (data processing)
- **Scikit-learn** (machine learning & preprocessing)
- **NumPy** (numerical operations)

---

## Dataset
The program expects a CSV file named **`car_data.csv`** with the following columns:

| Car Make | Car Model | Year | Mileage | Price | Color | Condition | Accident |
|----------|-----------|------|---------|-------|-------|-----------|----------|
| Audi     | A3        | 2019 | 25000   | 30000 | Grey  |   Used    |    No    |



## How to Run
1. Clone this repository or copy the script into a Python file, e.g., `car_price_predictor.py`.
2. Install dependencies:
   ```bash
   pip install pandas scikit-learn numpy

## Example Usage:
~~~
Enter car details for price prediction:
Available makes: ['aston martin', 'audi', 'bentley' ...]
Enter car make: audi
Available models: ['3 SERIES', '300', '488 GTB'...]
Enter car model: A3
Available years for A3: [np.int64(2010), np.int64(2011), np.int64(2012) ...
Enter car year: 2019
Available colors: ['BLACK', 'BLUE', 'BROWN', 'GRAY', 'GREEN', 'ORANGE', 'RED', 'SILVER', 'WHITE', 'YELLOW']
Enter car color: blue
Enter car mileage (minimum 20000 for higher accuracy outputs): 30000
Available conditions: ['like new', 'new', 'used']
Enter car condition: used
Available accident statuses: ['NO', 'YES']
Enter accident history: no

Predicted car price: $34,302.79

Would you like to predict another car price? (yes/no): no

Process finished with exit code 0
~~~

## Model Performance:
~~~
MSE: 777529165.57
MAE: 16373.58
R2 score: 0.81
Accuracy: 81.0%
~~~

## Future Improvements:
- Add a GUI (Graphical User Interface) for easier interaction.
- Improve dataset with more real-world car data.
- Experiment with other models (XGBoost, Gradient Boosting).
- Make a Neural Network model that can have higher accuracy
- Deploy as a web app (Flask/Django + React).
