import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

# Load the CSV file
df = pd.read_csv('housing-prices-dataset.csv')

# One-hot encode categorical columns
categorical_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
df = pd.get_dummies(df, columns=categorical_columns)

# Calculate the train size
train_size = int(0.7 * len(df))

# Split the DataFrame
df_train = df.iloc[:train_size]
df_test = df.iloc[train_size:]

# Prepare the data for training
X_train = df_train.drop('price', axis=1)
y_train = df_train['price']

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Prepare the data for prediction
X_test = df_test.drop('price', axis=1)

# Predict the prices
predicted_prices = model.predict(X_test)

# Create a copy of df_test and add the predicted prices as a new column
df_test_copy = df_test.copy()
df_test_copy.loc[:, 'predicted price'] = predicted_prices

# Concatenate the DataFrames
df = pd.concat([df_train, df_test_copy])

# Save the DataFrame
df.to_csv('housing_prices_predict.csv', index=False)