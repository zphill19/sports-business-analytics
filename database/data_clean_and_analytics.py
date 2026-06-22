import os
import sqlite3
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

print("🚀 ENGAGING PHASE 2: ADVANCED MACHINE LEARNING & GRIDSEARCHCV PIPELINE")

# --- ENVIRONMENT & ENVIRONMENT ROUTING LAYERS ---
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
root_dir = current_dir

# If executed from inside notebooks or database subfolders, snap back to root
if "notebooks" in current_dir or "database" in current_dir:
    root_dir = os.path.dirname(current_dir)

# Establish strict absolute file tracking paths
db_path = os.path.join(root_dir, "database", "sports_analytics.db")
csv_path = os.path.join(root_dir, "data", "raw_stadium_data.csv")

# --- DATA STORAGE VERIFICATION & INPUT CONSUMPTION ---
if not os.path.exists(csv_path):
    raise FileNotFoundError(
        f"\n[CRITICAL ERROR] Pipeline cannot locate Phase 1 data export at:\n{csv_path}\n"
        f"Please run your Phase 1 data generation script first to seed the repository!"
    )

print(f"[INFO] Secure connection established to data source: {csv_path}")
df = pd.read_csv(csv_path)

# 2. DATA CLEANING & TYPE MITIGATION LAYER (BLIND & MARKET-AWARE)
# =====================================================================
print("[INFO] Initiating Type Mitigation and Data Cleaning Pipelines...")

# ---------------------------------------------------------------------
# STEP 2A: TYPE MITIGATION (Standardizing Mixed Structural Variants)
# ---------------------------------------------------------------------
print("[INFO] Normalizing mixed string data types into pure numerical formats...")

# Force strict numerical coercion on numbers
# errors='coerce' turns accidental strings, text typos, or text 'NULL' entries into true NaNs
numeric_cols = ['clean_total_revenue', 'rain_chance_pct', 'knights_win_pct_10d', 'game_outcome']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Map mixed boolean text variants ('TRUE', '1.0', 'FALSE', '0.0') cleanly into integers (1 or 0)
binary_mappings = {
    'is_home_game': {'true': 1, '1.0': 1, '1': 1, 'false': 0, '0.0': 0, '0': 0},
    'has_promotion': {'true': 1, '1.0': 1, '1': 1, 'false': 0, '0.0': 0, '0': 0},
    'is_weekend': {'true': 1, '1.0': 1, '1': 1, 'false': 0, '0.0': 0, '0': 0}
}

for col, mapping in binary_mappings.items():
    # Convert safely to string, lowercase it, remove whitespace, and map to 1/0
    df[col] = df[col].astype(str).str.lower().str.strip()
    df[col] = df[col].map(mapping).fillna(0).astype(int)

print("[SUCCESS] Mixed data types successfully coerced to strict numeric/binary formats.")

# ---------------------------------------------------------------------
# STEP 2B: STATISTICAL CLEANING (Market-Segmented Anomaly Correction)
# ---------------------------------------------------------------------
print("[INFO] Scanning historical data for anomalies using market-segmented IQR rules...")

# Establish historical mask dynamically by tracking rows where game outcomes are known
is_historical = df['game_outcome'].notna()

# Isolate historical home games with positive revenue to build pristine financial profiles
homedata_clean = df[is_historical & (df['is_home_game'] == 1) & (df['clean_total_revenue'] > 0)]

# Pre-calculate medians, ceilings, and floors for each opponent market size dynamically
market_profiles = {}
for market in ['Large', 'Mid', 'Low']:
    market_rev = homedata_clean[homedata_clean['opponent_market_size'] == market]['clean_total_revenue']
    
    if not market_rev.empty:
        q1 = market_rev.quantile(0.25)
        q3 = market_rev.quantile(0.75)
        iqr = q3 - q1
        market_profiles[market] = {
            'median': market_rev.median(),
            'upper_bound': q3 + (1.5 * iqr),  
            'lower_bound': q1 - (1.5 * iqr) 
        }

# Process the historical segment row-by-row to detect and self-heal data anomalies
for idx, row in df[is_historical].iterrows():
    market = row['opponent_market_size']
    revenue = row['clean_total_revenue']
    is_home = row['is_home_game']
    
    # Fast-fail away games since they must legally have a revenue value of 0.0
    if is_home == 0:
        if revenue != 0.0 or pd.isna(revenue):
            df.at[idx, 'clean_total_revenue'] = 0.0
        continue
        
    # Skip processing if a market profile doesn't exist for the row's tier
    if market not in market_profiles:
        continue
        
    profile = market_profiles[market]
    
    # 1. Handle Missing Revenue Values (NaNs)
    if pd.isna(revenue):
        print(f"[ANOMALY FIXED] Game {row['game_id']} ({market} Market): Missing revenue imputed with Market Median (${profile['median']:,.2f})")
        df.at[idx, 'clean_total_revenue'] = profile['median']
        continue

    # 2. Handle Logical Boundary Violations (Negative Values)
    if revenue < 0:
        print(f"[ANOMALY FIXED] Game {row['game_id']} ({market} Market): Negative revenue clamped to $0.00")
        df.at[idx, 'clean_total_revenue'] = 0.0
        continue

    # 3. Handle Bidirectional Spikes (Upper Multipliers and Lower Data Drops)
    if revenue > profile['upper_bound']:
        print(f"[ANOMALY FIXED] Game {row['game_id']} ({market} Market): Extreme upper spike (${revenue:,.2f}) normalized to Market Median.")
        df.at[idx, 'clean_total_revenue'] = profile['median']
        
    elif revenue < profile['lower_bound'] and revenue > 0:
        print(f"[ANOMALY FIXED] Game {row['game_id']} ({market} Market): Extreme lower typo (${revenue:,.2f}) raised to Market Median.")
        df.at[idx, 'clean_total_revenue'] = profile['median']

print("[SUCCESS] Market-aware cleaning complete. All pipeline inputs perfectly stabilized.")

# =====================================================================
# 3. FUTURE METRIC GENERATION LAYER (PRE-GAME FEATURES ONLY)
# =====================================================================
print("[INFO] Simulating pre-game feature variables for the future timeline...")

# Lock the random seed to guarantee identical generation runs every time
np.random.seed(42)

# Establish masks to isolate past versus future timelines
is_historical = df['game_outcome'].notna()
is_future = df['game_outcome'].isna()

# --- STEP 3A: SIMULATE WEATHER CONDITIONS (GAMES 244+) ---
# Populate missing rain forecasts using an operational Beta distribution profile
future_indices = df[is_future].index
df.loc[is_future, 'rain_chance_pct'] = [round(np.random.beta(a=0.5, b=2.0), 2) for _ in range(len(future_indices))]

# --- STEP 3B: CHRONOLOGICAL GAME OUTCOMES & ROLLING WIN PERCENTAGE LOOPS ---
# Maintain an active list of game results to update team form dynamically row-by-row
simulated_outcomes = list(df.loc[is_historical, 'game_outcome'].values)

print("[INFO] Simulating chronological team form trajectories...")
for idx in future_indices:
    # 1. Calculate Team Form dynamically using the preceding 10-game window
    recent_window = simulated_outcomes[-10:]
    current_form = round(sum(recent_window) / len(recent_window), 2) if recent_window else 0.50
    df.at[idx, 'knights_win_pct_10d'] = current_form
    
    # 2. Simulate On-Field Win/Loss Results (Slight home field advantage boost)
    is_home = int(df.at[idx, 'is_home_game'])
    win_probability = 0.54 if is_home == 1 else 0.46
    sim_outcome = int(np.random.choice([0, 1], p=[1 - win_probability, win_probability]))
    
    df.at[idx, 'game_outcome'] = sim_outcome
    
    # 3. Append the result so subsequent rows see the updated performance trend
    simulated_outcomes.append(sim_outcome)

    # 4. Re-calculate a pristine historical rolling 10-game baseline using CLEAN revenues
df['rolling_avg_10d'] = (
    df.groupby('is_home_game')['clean_total_revenue']
    .transform(lambda x: x.shift(1).rolling(window=10, min_periods=1).mean())
    .fillna(150000.0)
    .round(2)
)
# Force away games to 0
df.loc[df['is_home_game'] == 0, 'rolling_avg_10d'] = 0.0

print("[SUCCESS] All pre-game classifier features completely filled for the entire season.")

# =====================================================================
# 4. EXPORT COMPLIANT CLEAN DATASET
# =====================================================================
# Construct the export path for the clean dataset
clean_csv_path = os.path.join(root_dir, "data", "clean_stadium_data.csv")

print(f"[INFO] Exporting prepared dataset to: {clean_csv_path}")

# Save the entire DataFrame containing clean historical rows and 
# simulated future features. Index=False keeps the CSV clean.
df.to_csv(clean_csv_path, index=False)

print("[SUCCESS] Phase 2 Data Preparation stage complete.")
print("'clean_stadium_data.csv' is secure and ready for Machine Learning inputs.")

# =====================================================================
# 5. MODEL TRAINING AND HYPERPARAMETER TUNING
# =====================================================================
df = pd.read_csv(clean_csv_path)
feature_cols = ['opponent_market_encoded', 'is_home_game', 'is_weekend', 'has_promotion', 'knights_win_pct_10d', 'rain_chance_pct']

#Isolate historical block (Games 1-243)
df_historical = df[df['game_id'] <= 243].copy()
df_future = df[df['game_id'] > 243].copy()

# Create a clean train/test cutoff entirely within the historical era
# 70% of historical data goes to Training, 15% goes to Validation, and 15% goes to Testing
total_historical = len(df_historical)
train_end = int(total_historical * 0.70)
val_end = train_end + int(total_historical * 0.15)

# Chronologically slice the historical segment
train_set = df_historical.iloc[:train_end]
val_set = df_historical.iloc[train_end:val_end]
test_set = df_historical.iloc[val_end:]

# Feature and Target matrices for training and local tuning
X_train, y_train = train_set[feature_cols], train_set['clean_total_revenue']
X_val, y_val = val_set[feature_cols], val_set['clean_total_revenue']
X_test, y_test = test_set[feature_cols], test_set['clean_total_revenue']

# Blind production matrix for games 244-324
X_forecast = df_future[feature_cols]
print(f"[INFO] Total Historical Records Available: {total_historical}")
print(f"       ->  Train Split (70%):      Games 1 to {train_end}   ({len(X_train)} records)")
print(f"       ->  Validation Split (15%): Games {train_end + 1} to {val_end}  ({len(X_val)} records)")
print(f"       ->  Test Split (15%):       Games {val_end + 1} to 243  ({len(X_test)} records)")
print(f"[INFO]  Production Forecast Segment:  Games 244 to 324 ({len(X_forecast)} records)")

# Map out cross-validation tracking
split_indecies = np.zeros(len(X_train) + len(X_val))
split_indecies[:len(X_train)] = -1
split_indecies[len(X_train):] = 0
custom_cv = PredefinedSplit(test_fold=split_indecies)

# Package pools for the grid container
X_tuning_pool = pd.concat([X_train, X_val], axis=0)
y_tuning_pool = pd.concat([y_train, y_val], axis=0)

# Parameter grid
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [4, 6, 8 , None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [2, 4, 6]
}

rf_base = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    cv=custom_cv,
    scoring='neg_mean_absolute_error',
    n_jobs=1
)

grid_search.fit(X_tuning_pool, y_tuning_pool)
best_model = grid_search.best_estimator_

print("\n" + "="*50 + "\nGRIDSEARCHCV TUNING PHASE METRICS\n" + "="*50)
print(f"Optimal Hyperparameters: {grid_search.best_params_}")
print(f"Best Validation Score:   ${abs(grid_search.best_score_):,.2f} MAE")
print("="*50 + "\n")

# Construct a strict absolute path to store the model
model_export_path = os.path.join(root_dir, "database", "revenue_rf_regressor.pkl")

print(f"[INFO] Freezing and saving trained model state to: {model_export_path}")
# Serialize and save the model structure to disk
joblib.dump(best_model, model_export_path)
print("[SUCCESS] Machine learning model state permanently secured.")
# =====================================================================
# 6. FINAL UNBIASED EVALUATION ON THE TEST SET
# =====================================================================

test_predictions = best_model.predict(X_test)
test_mae = mean_absolute_error(y_test, test_predictions)
test_r2 = r2_score(y_test, test_predictions)

print("="*50 + "\nFINAL UNBIASED TEST REPORT (HISTORICAL ERA)\n" + "="*50)
print(f"Final Test Mean Absolute Error: ${test_mae:,.2f}")
print(f"Final Test R-Squared Score:     {test_r2:.4f}")
print("="*50 + "\n")

# =====================================================================
# 7. PRODUCTION FORECASTING AND FEATURE IMPORTANCE
# =====================================================================
print("[INFO] Deploying tuned regressor to predict expected future revenues...")

# Apply winning model to predict the future revenue rows
df_future['clean_total_revenue'] = np.round(best_model.predict(X_forecast), 2)
df_future.loc[df_future['is_home_game'] == 0, 'clean_total_revenue'] = 0.0

# Re-combine timelines chronologically to safely compute rolling indicators
df_simulated = pd.concat([df_historical, df_future], axis=0).sort_values('game_id').reset_index(drop=True)

print("[INFO] Re-computing true operational rolling baselines and target flags...")
# Re-calculate moving 10-day averages using clean + predicted values
df_simulated['rolling_avg_10d'] = (
    df_simulated.groupby('is_home_game')['clean_total_revenue']
    .transform(lambda x: x.shift(1).rolling(window=10, min_periods=1).mean())
    .round(2)
)
df_simulated['target_above_avg'] = np.nan

# Dynamically extract binary flags from continuous outputs
df_simulated.loc[(df_simulated['is_home_game'] == 1) & (df_simulated['clean_total_revenue'] < df_simulated['rolling_avg_10d']), 'target_above_avg'] = 0
df_simulated.loc[(df_simulated['is_home_game'] == 1) & (df_simulated['clean_total_revenue'] >= df_simulated['rolling_avg_10d']), 'target_above_avg'] = 1
df_simulated.loc[df_simulated['is_home_game'] == 0, 'rolling_avg_10d'] = np.nan
df_simulated.loc[df_simulated['is_home_game'] == 0, 'target_above_avg'] = np.nan

print("\n" + "="*50 + "\nFEATURE IMPORTANCE RANKINGS (BUSINESS INSIGHTS)\n" + "="*50)
    
# Extract the importance values from the winning forest architecture
importances = best_model.feature_importances_
    
# Sort feature indices in descending order based on importance scores
indices = np.argsort(importances)[::-1]
    
# Print out the clean ranked breakdown to the terminal
for rank in range(X_train.shape[1]):
    feature_name = feature_cols[indices[rank]]
    importance_score = importances[indices[rank]]
    print(f"{rank + 1}. {feature_name:<25} -> {importance_score:.4f} ({importance_score*100:.1f}%)")
        
print("="*50 + "\n")

# =====================================================================
# 8. REPOSITORY INJECTION LAYER
# =====================================================================
print("[INFO] Uploading production machine learning matrices back to database...")

# Format fields cleanly for table consumption
df_production_predictions = df_simulated[df_simulated['game_id'] > 243][
    ['game_id', 'clean_total_revenue', 'rolling_avg_10d', 'target_above_avg']
].copy()

df_production_predictions.columns = [
    'game_id', 'predicted_revenue', 'predicted_rolling_avg_10d', 'predicted_target_above_avg'
]

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Commit to fresh SQL infrastructure table
cursor.execute("DROP TABLE IF EXISTS production_predictions;")
df_production_predictions.to_sql("production_predictions", conn, if_exists="append", index=False)
conn.commit()

# Sample Verification Preview
check_df = pd.read_sql_query("SELECT * FROM production_predictions LIMIT 5", conn)
conn.close()

print(f"[SUCCESS] Phase 2 operations completely secure.")
print(f"Predictions successfully written to table inside: {db_path}")
print("\n PRODUCTION FORECAST SEGMENT PREVIEW (GAMES 244+):")
print(check_df.to_string(index=False))
