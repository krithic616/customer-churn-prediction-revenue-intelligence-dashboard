from src.data_preprocessing import load_data
from src.feature_engineering import create_features
from src.train_model import train_model

print("Step 1: Loading data...")
df = load_data()

print("Step 2: Feature engineering...")
df = create_features(df)

print("Step 3: Training model...")
train_model(df)

print("Pipeline completed successfully!")
