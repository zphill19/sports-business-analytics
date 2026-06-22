import sqlite3
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 1. DIRECTORY RESOLUTION & ENVIRONMENT INITIALIZATION
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
root_dir = current_dir
if "notebooks" in current_dir or "database" in current_dir:
    root_dir = os.path.dirname(current_dir)

db_dir = os.path.join(root_dir, "database")
data_dir = os.path.join(root_dir, "data")

for folder in [db_dir, data_dir]:
    if not os.path.exists(folder):
        os.makedirs(folder)

db_path = os.path.join(db_dir, "sports_analytics.db")
csv_path = os.path.join(data_dir, "raw_stadium_data.csv")

print("   ENGAGING PHASE 1: GENERATING PORTFOLIO RAW SYSTEM EXPORT")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 2. DATA DEFINITION LAYER (DDL) - STAR SCHEMA SETUP
cursor.execute("DROP VIEW IF EXISTS v_analytics_presentation_layer;")
cursor.execute("DROP TABLE IF EXISTS fact_stadium_transactions;")
cursor.execute("DROP TABLE IF EXISTS fact_schedule;")
cursor.execute("DROP TABLE IF EXISTS dim_calendar;")
cursor.execute("DROP TABLE IF EXISTS dim_teams;")

cursor.execute("""
CREATE TABLE dim_teams (
    team_id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_name TEXT UNIQUE NOT NULL,
    market_size TEXT NOT NULL,
    market_encoded INTEGER NOT NULL
);
""")

cursor.execute("""
CREATE TABLE dim_calendar (
    date_id TEXT PRIMARY KEY,
    calendar_year INTEGER NOT NULL,
    month_number INTEGER NOT NULL,
    day_name TEXT NOT NULL,
    is_weekend INTEGER NOT NULL
);
""")

cursor.execute("""
CREATE TABLE fact_schedule (
    game_id INTEGER PRIMARY KEY AUTOINCREMENT,
    date_id TEXT NOT NULL,
    season_year INTEGER NOT NULL,
    game_number INTEGER NOT NULL,
    series_number INTEGER NOT NULL,
    opponent_id INTEGER NOT NULL,
    is_home_game TEXT NOT NULL,         
    has_promotion TEXT NOT NULL,        -- Known ahead of time due to marketing calendars
    rain_chance_pct REAL,             
    game_outcome INTEGER,            
    knights_win_pct_10d REAL NOT NULL,   
    FOREIGN KEY (date_id) REFERENCES dim_calendar(date_id),
    FOREIGN KEY (opponent_id) REFERENCES dim_teams(team_id)
);
""")

cursor.execute("""
CREATE TABLE fact_stadium_transactions (
    transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER NOT NULL,
    raw_total_revenue REAL,            
    FOREIGN KEY (game_id) REFERENCES fact_schedule(game_id)
);
""")

print("[SUCCESS 1/4] Relational star schema warehouse layouts initialized.")

# 3. LEAGUE TIMELINE SEEDING ENGINE
large_market_teams = ["Knights", "Waves", "Wind", "Beacons", "Fog", "Stampede", "Bells", "Cosmos", "Senators", "Towers"]
mid_market_teams   = ["Fire", "Sun", "Sandstorms", "Millers", "Motors", "Emeralds", "Peaks", "Surf", "Tides", "Archers"]
low_market_teams   = ["Steamers", "Porkers", "Rivers", "Monarchs", "Brew", "Clippers", "Oaks", "Neon", "Lumber", "Bees"]

teams_data = []
for idx, tier in enumerate([large_market_teams, mid_market_teams, low_market_teams], start=1):
    market_str = {1: 'Large', 2: 'Mid', 3: 'Low'}[idx]
    weight     = {1: 3, 2: 2, 3: 1}[idx]
    for team in tier:
        teams_data.append((team, market_str, weight))

df_temp_teams = pd.DataFrame(teams_data, columns=['team_name', 'market_size', 'market_encoded'])
df_temp_teams.to_sql("dim_teams", conn, if_exists="append", index=False)

df_dim_teams = pd.read_sql_query("SELECT * FROM dim_teams", conn)

opponents_pool = df_dim_teams[df_dim_teams['team_name'] != 'Knights'].copy()
print("[SUCCESS 2/4] League dimension tracking tables populated (Our Team = Knights).")

np.random.seed(42)
season_start_dates = {2025: datetime(2025, 3, 27), 2026: datetime(2026, 3, 26)}
calendar_records = set()
schedule_records = []
transaction_records = []

global_game_counter = 1

for season in [2025, 2026]:
    home_series_flags = np.random.permutation([1]*27 + [0]*27)
    series_opponents_indices = np.random.choice(len(opponents_pool), size=54, replace=True)
    days_elapsed = 0
    season_game_outcomes = []
    
    for series_idx, is_home in enumerate(home_series_flags):
        series_num = series_idx + 1
        opp_row = opponents_pool.iloc[series_opponents_indices[series_idx]]
        opp_id = int(opp_row['team_id'])
        opp_market = opp_row['market_size']
        
        series_has_promo = 0
        promo_game_target = -1
        
        if is_home:
            if opp_market in ['Mid', 'Low']:
                series_has_promo = int(np.random.choice([0, 1], p=[0.3, 0.7]))
            else:
                series_has_promo = int(np.random.choice([0, 1], p=[0.8, 0.2]))
                
            if series_has_promo:
                promo_game_target = np.random.choice([1, 2, 3])
        
        for game_in_series in range(1, 4):
            game_num = (series_idx * 3) + game_in_series
            current_date = season_start_dates[season] + timedelta(days=days_elapsed)
            date_str = current_date.strftime('%Y-%m-%d')
            is_wknd = 1 if current_date.weekday() in [4, 5, 6] else 0
            
            calendar_records.add((date_str, current_date.year, current_date.month, current_date.strftime('%A'), is_wknd))
            
            messy_home_str = "TRUE" if is_home else "0.0"
            messy_promo_str = "1.0" if (is_home and game_in_series == promo_game_target) else "FALSE"
            
            # Extract current win percentages up to this specific point in time
            if len(season_game_outcomes) == 0:
                current_win_pct = 0.50
            else:
                recent_games = season_game_outcomes[-10:]
                current_win_pct = round(sum(recent_games) / len(recent_games), 2)
            
            # --- APPLY TIME WALL SEMANTICS ---
            if global_game_counter <= 243:
                # Historical Segment: Weather records and game outcomes are completely known
                rain_chance = round(np.random.beta(a=0.5, b=2.0), 2)
                win_probability = 0.54 if is_home else 0.46
                simulated_outcome = int(np.random.choice([0, 1], p=[1 - win_probability, win_probability]))
                
                # Financial data calculation
                if is_home:
                    base_noise = np.random.normal(1.0, 0.08)
                    promo_noise = np.random.normal(1.25, 0.05) if (is_home and game_in_series == promo_game_target) else 1.0
                    weather_noise = np.random.normal(1.0, 0.12) * (1.0 - (rain_chance * 0.40))
                    perf_noise = np.random.normal(1.0, 0.04) * (1.0 + ((current_win_pct - 0.50) * 0.30))
                    
                    market_mult = {'Large': 1.6, 'Mid': 1.1, 'Low': 0.75}[opp_market]
                    weekend_mult = 1.4 if is_wknd else 1.0
                    
                    total_multiplier = market_mult * weekend_mult * promo_noise * weather_noise * perf_noise * base_noise
                    revenue = round(np.random.normal(280000, 35000) * total_multiplier, 2)
                else:
                    revenue = 0.0
                
                schedule_records.append((date_str, season, game_num, series_num, opp_id, messy_home_str, messy_promo_str, rain_chance, simulated_outcome, current_win_pct))
                transaction_records.append((revenue,))
                season_game_outcomes.append(simulated_outcome)
            else:
                # Future Segment: Weather forecasts, game outcomes, and POS cash fields are totally blank (None)
                # Keep current win percentage as our static baseline anchor before games play out
                schedule_records.append((date_str, season, game_num, series_num, opp_id, messy_home_str, messy_promo_str, None, None, current_win_pct))
                transaction_records.append((None,))
            
            global_game_counter += 1
            days_elapsed += 1
        days_elapsed += 1

# Populate Database Tables
pd.DataFrame(list(calendar_records), columns=['date_id', 'calendar_year', 'month_number', 'day_name', 'is_weekend']).to_sql("dim_calendar", conn, if_exists="append", index=False)
pd.DataFrame(schedule_records, columns=['date_id', 'season_year', 'game_number', 'series_number', 'opponent_id', 'is_home_game', 'has_promotion', 'rain_chance_pct', 'game_outcome', 'knights_win_pct_10d']).to_sql("fact_schedule", conn, if_exists="append", index=False)

df_fact_trans = pd.DataFrame(transaction_records, columns=['raw_total_revenue'])

# Inject deliberate database bugs strictly inside the historic segments (games 1 to 243)
home_indices = df_fact_trans[(df_fact_trans['raw_total_revenue'] > 0) & (df_fact_trans.index < 243)].index
df_fact_trans.loc[np.random.choice(home_indices, size=int(len(home_indices)*0.04), replace=False), 'raw_total_revenue'] = np.nan
df_fact_trans.loc[np.random.choice(home_indices, size=int(len(home_indices)*0.02), replace=False), 'raw_total_revenue'] = -9500.00
spike_indices = np.random.choice(home_indices, size=int(len(home_indices)*0.015), replace=False)
df_fact_trans.loc[spike_indices, 'raw_total_revenue'] = df_fact_trans.loc[spike_indices, 'raw_total_revenue'] * 100.0

df_fact_trans.insert(0, 'game_id', range(1, len(df_fact_trans) + 1))
df_fact_trans.to_sql("fact_stadium_transactions", conn, if_exists="append", index=False)

# 4. TRANSFORMATION LAYER VIEW (ELT ARCHITECTURE)
cursor.execute("""
CREATE VIEW v_analytics_presentation_layer AS
WITH compiled_base AS (
    SELECT 
        fs.game_id,
        fs.date_id AS game_date,
        fs.season_year,
        fs.game_number,
        fs.series_number,
        dt.team_name AS opponent_team_name,
        dt.market_size AS opponent_market_size,
        dt.market_encoded AS opponent_market_encoded,
        CASE WHEN fs.is_home_game IN ('TRUE', '1.0', '1') THEN 1 ELSE 0 END AS is_home_game,
        dc.is_weekend,
        CASE WHEN fs.has_promotion IN ('TRUE', '1.0', '1') THEN 1 ELSE 0 END AS has_promotion,
        fs.rain_chance_pct,
        fs.game_outcome,
        fs.knights_win_pct_10d,
        CASE WHEN ft.raw_total_revenue < 0 THEN 0.0
                    WHEN ft.raw_total_revenue IS NULL AND fs.game_id <= 243 THEN 150000.0 ELSE ft.raw_total_revenue END AS clean_total_revenue
        FROM fact_schedule fs
        JOIN dim_teams dt ON fs.opponent_id = dt.team_id
        JOIN dim_calendar dc ON fs.date_id = dc.date_id
        JOIN fact_stadium_transactions ft ON fs.game_id = ft.game_id
        ),
        rolling_windows AS (
        SELECT *,
        AVG(clean_total_revenue) OVER (PARTITION BY is_home_game
                    ORDER BY game_id ASC ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING)
        AS rolling_avg_10d
        FROM compiled_base
        )
        SELECT game_id, game_date, season_year, game_number, series_number, opponent_team_name, opponent_market_size, opponent_market_encoded, is_home_game, is_weekend, has_promotion, rain_chance_pct, game_outcome, knights_win_pct_10d, clean_total_revenue,
        CASE WHEN is_home_game AND game_id <= 243 THEN rolling_avg_10d ELSE 0.0 END AS rolling_avg_10d,
        CASE WHEN clean_total_revenue >= rolling_avg_10d AND is_home_game = 1 AND game_id <= 243 THEN 1 ELSE 0 END AS target_above_avg FROM rolling_windows
        """)
conn.commit()

# =====================================================================
# 5. EXPORT CONSTRAINED SYSTEM SNAPSHOT TO CSV FOR GITHUB
# =====================================================================

df_csv_export = pd.read_sql_query("SELECT * FROM v_analytics_presentation_layer ORDER BY game_id ASC", conn)
df_csv_export.to_csv(csv_path, index=False)
conn.close()

print(f"\n[SUCCESS] Phase 1 Real-World Data Structure Secure.")
print(f"File successfully compiled and exported to repository path: {csv_path}")
print("Games 244-324 have had unknown future variables programmatically wiped.")
