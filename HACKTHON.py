#Step 1 Setting working directory and importing python
import os
os.chdir('C:/Users/dipen/OneDrive/Desktop/Adelphi University/MS BUSINESS ANALYTICS/Applied Machine Learning/DSC 681 Hackthon Project Data')
import pandas as pd

# Step 2.Load datasets
tweet_info = pd.read_csv('tweet_info.csv')
tweet_more_info = pd.read_csv('tweet_more_info.txt', delimiter='\t')  # Assuming tab-separated file
user_profile = pd.read_csv('user_profile.csv')

# Step 3 Check for missing values
print(tweet_info.isnull().sum())
print(tweet_more_info.isnull().sum())
print(user_profile.isnull().sum())

#Step 4. Remove missing values and duplicates
tweet_info = tweet_info.dropna(subset=['text', 'quote'])
user_profile = user_profile.dropna(subset=['name', 'followers_num','tweets_num', 'following_num', 'location', 'join_date', 'user_desc'])

# Step 5. Show the first few rows of each dataset
print(tweet_info.head())
print(tweet_more_info.head())
print(user_profile.head())

#Step 6. Check for duplicates in the DataFrames
# Filter and display the duplicate rows from the 'tweet_info' DataFrame
print(tweet_info.duplicated().sum())
print(tweet_more_info.duplicated().sum())
print(user_profile.duplicated().sum())


# Step 7: Consolidate the data
# Merge datasets based on 'tweet_id' and 'username'
merged_data = tweet_info.merge(tweet_more_info, on='tweet_id', how='left')
merged_data = merged_data.merge(user_profile, on='username', how='left')

# Step 9: Drop unnecessary columns
merged_data = merged_data.drop(columns=['text', 'quote', 'location', 'user_desc'])

# Step 10: Fill missing values and check data 
print(merged_data.isnull().sum()) 


# Fill NaN values for specific columns 
columns_to_fill = { 
    'tweets_num': 0, 
    'following_num': 0, 
    'followers_num': 0, 
    'character': 0, 
    'term': 0, 
    'hashtag': 0, 
    'mention': 0, 
    'name': 'Unknown', 
    'join_date': 'Unknown' 
    } 

# Apply fillna() for each column 
for column, value in columns_to_fill.items():
    merged_data[column] = merged_data[column].fillna(value)
    
# Check for NaN values after filling 
print("After filling NaNs:")
print(merged_data.isnull().sum())

# Step 11: Split the data into dependent (y) and independent (X) variables
# 'rt_num' is the dependent variable, and other columns are independent variables(features)
y = merged_data['rt_num']
X = merged_data[['hashtag', 'img_num', 'has_vid', 'mention', 'followers_num', 'tweets_num']]

# Calculate statistics 
stats = merged_data[['rt_num', 'hashtag', 'img_num', 'has_vid', 'mention', 'followers_num', 'tweets_num']].describe().transpose() 

# Rename columns for clarity 
stats.columns = ['Number of Observations', 'Min',  '1st Quartile', 'Median', 'Mean','3rd Quartile','Max', 'Standard Deviation'] 

# Add standard deviation column 
stats['Standard Deviation'] = merged_data[['rt_num', 'hashtag', 'img_num', 'has_vid', 'mention', 'followers_num', 'tweets_num']].std()
stats

# Step 12: Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train
y_train

# Standardize the features 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Debug: Print shapes of training and test sets 
print(f"X_train shape: {X_train.shape}") 
print(f"y_train shape: {y_train.shape}") 
print(f"X_test shape: {X_test.shape}") 
print(f"y_test shape: {y_test.shape}")
print(X_train.var(axis=0))  # Variance of each feature 

from sklearn.svm import SVR
svm_model = SVR(kernel='linear', C=0.1) # Using linear Kernel 
X_train_sample = X_train[:100000]  # Use a smaller sample
y_train_sample = y_train[:100000]
svm_model.fit(X_train_sample,y_train_sample)

from sklearn.svm import LinearSVR
from sklearn.metrics import r2_score
linear_svr = LinearSVR(C=0.1, max_iter=1000)
linear_svr.fit(X_train, y_train)
y_pred = linear_svr.predict(X_test)
print(f"R2 Score:{r2_score(y_test,y_pred)}")

# Step 15: Evaluate the model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test, y_pred) 
mse = mean_squared_error(y_test, y_pred) 
r2 = r2_score(y_test, y_pred) 
print("\nModel Evaluation:") 
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}") 
print(f"R2 Score: {r2}")
