
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Dataset .csv')

print("--- Starting Data Preprocessing ---")

# Drop irrelevant columns
df = df.drop(['Restaurant ID', 'Restaurant Name', 'Country Code', 'City', 'Address', 'Locality', 'Locality Verbose', 'Longitude', 'Latitude', 'Currency', 'Rating color', 'Rating text', 'Switch to order menu'], axis=1)

# Handle missing values in 'Cuisines'
df['Cuisines'].fillna('No Cuisines', inplace=True)

# Encode categorical variables
le = LabelEncoder()
df['Has Table booking'] = le.fit_transform(df['Has Table booking'])
df['Has Online delivery'] = le.fit_transform(df['Has Online delivery'])
df['Is delivering now'] = le.fit_transform(df['Is delivering now'])

# One-hot encode 'Cuisines'
df = pd.get_dummies(df, columns=['Cuisines'], prefix='Cuisine')

# Split the data into features (X) and target (y)
X = df.drop('Aggregate rating', axis=1)
y = df['Aggregate rating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("--- Data Preprocessing Complete ---")

print("--- Training Model ---")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
print("--- Model Training & Evaluation Complete ---")


print("--- Generating Plot ---")
# Get feature importances
feature_importances = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
feature_importances = feature_importances.sort_values('importance', ascending=False).head(10)

# Plot the graph
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importances)
plt.title('Top 10 Feature Importances')
plt.show()


# Plot the graph
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importances)
plt.title('Top 10 Feature Importances')

plt.savefig('feature_importances.png') 

print("Plot has been saved as feature_importances.png")