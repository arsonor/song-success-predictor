import pickle

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE


# parameters

smote = SMOTE(random_state=42)
model = RandomForestClassifier(max_depth=30, max_features='log2', min_samples_leaf=1, min_samples_split=2, n_estimators=476, class_weight='balanced')
output_file = 'hit-model.bin'

# data preparation

df = pd.read_csv('../data/dataset_ready.csv')

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

X_full_train = df_full_train.drop(columns=['song_id', 'hit'])
y_full_train = df_full_train['hit']
X_train = df_train.drop(columns=['song_id', 'hit'])
y_train = df_train['hit']
X_val = df_val.drop(columns=['song_id', 'hit'])
y_val = df_val['hit']
X_test = df_test.drop(columns=['song_id', 'hit'])
y_test = df_test['hit']

numerical = ['release_year', 'duration_log', 'popularity', 'followers_log', 'acousticness', 'liveness', 'speechiness', 
            'instrumentalness', 'loudness', 'energy', 'danceability', 'valence', 'tempo']
categorical = ['key', 'artist_type', 'genre']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical),
        ('cat', OneHotEncoder(drop='first'), categorical)
    ]
)

# Validate the model

print(f'doing validation with rf')

X_train = preprocessor.fit_transform(X_train)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
model.fit(X_train_resampled, y_train_resampled)

X_val = preprocessor.transform(X_val)
y_pred = model.predict_proba(X_val)[:, 1]
print('Validation results:')
print(f"ROC-AUC: {roc_auc_score(y_val, y_pred):.4f}")


# Test the model

print('training the final model')

X_full_train = preprocessor.fit_transform(X_full_train)
X_full_train_resampled, y_full_train_resampled = smote.fit_resample(X_full_train, y_full_train)
model.fit(X_full_train_resampled, y_full_train_resampled)

X_test = preprocessor.transform(X_test)
y_pred = model.predict_proba(X_test)[:, 1]
print(f"Test ROC-AUC: {roc_auc_score(y_test, y_pred):.4f}")


# Save the preprocessor and model separately

with open(output_file, 'wb') as f_out:
    pickle.dump({
        'preprocessor': preprocessor,
        'model': model
    }, f_out)

print(f'The model and preprocessor are saved to {output_file}')