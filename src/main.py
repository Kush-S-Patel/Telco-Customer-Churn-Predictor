
import pandas as pd
import numpy as np
from scipy import stats


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data/data.csv")


if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])


df['Churn'] = df['Churn'].map({'Yes': True, 'No': False})

cols_le = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
cols_numeric = ['tenure', 'MonthlyCharges', 'TotalCharges']

# drop_cols = ['gender', 'TotalCharges', 'PhoneService', 'StreamingTV', 'StreamingMovies', 'InternetService']

# cols_le = [col for col in cols_le if col not in drop_cols]
# cols_numeric = [col for col in cols_numeric if col not in drop_cols]

target = 'Churn'

df_copy = df.copy()

df_copy[target] = df_copy[target].map({'Yes': 1, 'No': 0}).fillna(df_copy[target])

cat_cols = df_copy.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in cat_cols:
    df_copy[col] = le.fit_transform(df_copy[col].astype(str))

correlations = df_copy.corr(numeric_only=True)[target].sort_values(ascending=False)

plt.figure(figsize=(6, len(correlations)/3))
sns.heatmap(correlations.to_frame(), annot=True, cmap='coolwarm', center=0)
plt.title("Correlation of All Features with Churn")
plt.show()



print(df_copy['TotalCharges'].describe())
plt.figure(figsize=(9, 8))
# sns.histplot(df_copy['TotalCharges'], color='g', bins=100, hist_kws={'alpha': 0.4})

print(df_copy['tenure'].describe())
plt.figure(figsize=(9, 8))
# sns.histplot(df_copy['tenure'], color='g', bins=100, hist_kws={'alpha': 0.4})

print(df_copy['MonthlyCharges'].describe())
plt.figure(figsize=(9, 8))
# sns.histplot(df_copy['MonthlyCharges'], color='g', bins=100, hist_kws={'alpha': 0.4})
#highest positive correlations to Churn: monthlycharges, paperlessbilling, seniorcitizen
#highest negative correlations to Churn: Contract, tenure, onlinesecurity
#least useful categories: totalcharges, phoneservice,gender,streamingtv,streamingmovies,internetservice


df_encoded = pd.get_dummies(df, columns=cols_le)
print(df_encoded.columns)
# df_encoded = df_encoded.drop(columns=drop_cols, errors='ignore')
for col in cols_numeric:
    df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')

print(df_encoded)
# corr = df_encoded.corr(numeric_only=True)
# plt.figure(figsize=(12, 12))
# sns.heatmap(corr, annot=False, cmap='coolwarm')
# plt.title("Correlation Heatmap")
# plt.show()

num_cols = df_encoded.select_dtypes(include=['number']).columns
df_encoded[num_cols] = df_encoded[num_cols].interpolate(method='linear', limit_direction='both')

# Fill categorical (non-numeric) columns with forward fill then backward fill
cat_cols = df_encoded.select_dtypes(exclude=['number']).columns
df_encoded[cat_cols] = df_encoded[cat_cols].fillna(method='ffill').fillna(method='bfill')

z_scores = np.abs(stats.zscore(df_encoded.select_dtypes(include=['float64','int64'])))
df_encoded = df_encoded[(z_scores < 3).all(axis=1)]


X = df_encoded.drop(columns=['Churn'])
y = df_encoded['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Horizontal boxplots for each numerical column
for i, column in enumerate(cols_numeric):
    plt.subplot(3, 3, i+1)
    sns.boxplot(y=X_train[column])
    plt.title(column)


plt.tight_layout()
plt.show()

rf = RandomForestClassifier(n_estimators=500,
    max_depth=12,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
train_pred = rf.predict(X_train)
print("\nRandom Forest Classifier\n")
print("Training Accuracy:", accuracy_score(y_train, train_pred))
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))


model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nLinear Classifier\n")

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

print("\nXGB Classifier\n")

model = XGBClassifier(eval_metric='auc')

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
# print("Classification Report:")
# print(classification_report(y_test, y_pred))

print("\nGradient Boost Classifier\n")
gb_model = GradientBoostingClassifier(
    n_estimators=100,      # number of trees
    learning_rate=0.1,     # shrinkage
    max_depth=3,           # depth of each tree
    random_state=42
)

# Train
gb_model.fit(X_train, y_train)

# Predictions
y_pred_train = gb_model.predict(X_train)
y_pred_test = gb_model.predict(X_test)

# Evaluate
print("Train accuracy:", accuracy_score(y_train, y_pred_train))
print("Test accuracy:", accuracy_score(y_test, y_pred_test))
