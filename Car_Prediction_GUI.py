import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from sklearn.linear_model import LinearRegression
import pandas as pd
from pandas.core.frame import DataFrame

def predict_price():
    try:
        # Get user inputs
        levy = float(levy_entry.get())
        manufacturer = manufacturer_entry.get()
        prod_year = int(prod_year_entry.get())
        fuel_type = fuel_type_entry.get()
        engine_volume = float(engine_volume_entry.get())
        mileage = float(mileage_entry.get())
        airbags = int(airbags_entry.get())

        # Create DataFrame from user input
        selected_data = pd.DataFrame({
            'Levy': [levy],
            'Manufacturer': [manufacturer],
            'Prod. year': [prod_year],
            'Fuel type': [fuel_type],
            'Engine volume': [engine_volume],
            'Mileage': [mileage],
            'Airbags': [airbags]
        })

        # Read the CSV file into a pandas DataFrame
        car_data = pd.read_csv('car_price_prediction.csv')

        # Feature selection based on user input features
        selected_features = ['Price', 'Levy', 'Manufacturer', 'Prod. year', 'Fuel type', 'Engine volume', 'Mileage', 'Airbags']
        selected_data = car_data[selected_features]

        # Data conversion to numeric values
        numerical_data = pd.get_dummies(selected_data)

        # Separate user input data from other dataset
        input_data = numerical_data.tail(1).drop('Price', axis=1)

        # Load pre-trained model
        model = LinearRegression()
        X = numerical_data.drop('Price', axis=1)
        y = numerical_data['Price']
        model.fit(X, y)

        # Predict car price
        predicted_price = model.predict(input_data)[0]

        # Show prediction result
        messagebox.showinfo("Prediction Result", f"Predicted Car Price: ${predicted_price:.2f}")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Create main application window
root = tk.Tk()
root.title("Car Price Prediction")

# Create entry fields for user input
levy_label = ttk.Label(root, text="Levy:")
levy_label.grid(row=0, column=0, padx=10, pady=5)
levy_entry = ttk.Entry(root)
levy_entry.grid(row=0, column=1, padx=10, pady=5)

manufacturer_label = ttk.Label(root, text="Manufacturer:")
manufacturer_label.grid(row=1, column=0, padx=10, pady=5)
manufacturer_entry = ttk.Entry(root)
manufacturer_entry.grid(row=1, column=1, padx=10, pady=5)

prod_year_label = ttk.Label(root, text="Prod. Year:")
prod_year_label.grid(row=2, column=0, padx=10, pady=5)
prod_year_entry = ttk.Entry(root)
prod_year_entry.grid(row=2, column=1, padx=10, pady=5)

fuel_type_label = ttk.Label(root, text="Fuel Type:")
fuel_type_label.grid(row=3, column=0, padx=10, pady=5)
fuel_type_entry = ttk.Entry(root)
fuel_type_entry.grid(row=3, column=1, padx=10, pady=5)

engine_volume_label = ttk.Label(root, text="Engine Volume:")
engine_volume_label.grid(row=4, column=0, padx=10, pady=5)
engine_volume_entry = ttk.Entry(root)
engine_volume_entry.grid(row=4, column=1, padx=10, pady=5)

mileage_label = ttk.Label(root, text="Mileage:")
mileage_label.grid(row=5, column=0, padx=10, pady=5)
mileage_entry = ttk.Entry(root)
mileage_entry.grid(row=5, column=1, padx=10, pady=5)

airbags_label = ttk.Label(root, text="Airbags:")
airbags_label.grid(row=6, column=0, padx=10, pady=5)
airbags_entry = ttk.Entry(root)
airbags_entry.grid(row=6, column=1, padx=10, pady=5)

# Add predict button
predict_button = ttk.Button(root, text="Predict Price", command=predict_price)
predict_button.grid(row=7, columnspan=2, padx=10, pady=10)

# Run the main event loop
root.mainloop()
