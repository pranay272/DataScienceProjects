import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score


# Load the data
df = pd.read_csv("C:/Users/parab/OneDrive/Desktop/p3/walmart2.csv")

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'],format='%d-%m-%Y')

# Extract components
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['day_of_week'] = df['Date'].dt.dayofweek
df['season'] = (df['Date'].dt.month % 12 + 3) // 3

# Encode categorical features
season_mapping = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Autumn'}
df['season'] = df['season'].map(season_mapping)

le = LabelEncoder()
df['season'] = le.fit_transform(df['season'])
df['IsHoliday'] = le.fit_transform(df['IsHoliday'])

# Split the data into input features (X) and target variable (Y)
#X = df[['Store', 'Dept', 'IsHoliday', 'year', 'month', 'day', 'day_of_week', 'season']]
Y = df['Weekly_Sales']
X = df.drop('Weekly_Sales', axis = 1)
X = X.drop('Date', axis = 1)
print(X)
# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# Linear Regression
linR = LinearRegression()
linR.fit(X_train, Y_train)

# Make predictions
y_pred = linR.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)