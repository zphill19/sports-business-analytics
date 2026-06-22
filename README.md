#  Comprehensive Sports Business Analytics & Predictive Lifecycle Suite

##  Project Overview & Architecture
This repository contains a full-stack data engineering, predictive machine learning, and prescriptive business intelligence system. Because high-resolution sports venue transactions are heavily proprietary, **I designed, seeded, and engineered a custom synthetic sports analytics database from scratch** to model a 324-game operational cycle for a professional baseball team (The Knights).

The entire system is architected around an explicit production **Time-Wall Semantic boundary at Game 243**. This splits the pipeline into a messy historical era (Games 1–243) and an unplayed production forecasting era (Games 244–324) where future variables are completely unknown.

sports-business-analytics-lifecycle/
├── dashboards/
│   └── Stadium_Revenue_Forecast_Optimizer.pbix   # Phase 3: Prescriptive Analytics Console
├── data/
│   ├── raw_stadium_data.csv                       # Phase 1: Output System Snapshot
│   └── clean_stadium_data.csv                     # Phase 2: Sanitized Machine Learning Input
├── database/
│   ├── database_setup.py                          # Phase 1: Star Schema Database Seed Engine
│   ├── sports_analytics.db                        # Relational SQL Infrastructure (SQLite)
│   ├── revenue_rf_regressor.pkl                   # Frozen/Serialized ML Model (Joblib)
│   └── data_clean_and_analytics.py                # Phase 2: Advanced ETL & Machine Learning Pipeline
├── executive_summary.md                           # Phase 4: Standing Business Memo
└── README.md     

---

##  Phase 1: Local SQL Infrastructure & Star Schema Seeding
The data lifecycle begins by executing `phase1_data_generation.py`, which instantiates a local SQLite database (`sports_analytics.db`) using strict relational data definitions (DDL) and seeds five primary entities:
*   `dim_teams`: Tracks 30 league opponents stratified across **Large, Mid, and Low market sizes** to capture baseline market pull.
*   `dim_calendar`: A normalized calendar index routing seasonal game dates by day of the week, month, and weekend status.
*   `fact_schedule`: Holds game-day parameters (`rain_chance_pct`, marketing `has_promotion`, and rolling 10-day `knights_win_pct_10d` team form).
*   `fact_stadium_transactions`: Records granular stadium dollar entries split out by ticket gates, concessions, and merchandise streams.

###  Deliberate Production Bug Injection
To simulate real-world data degradation, the generation engine introduces three deliberate anomalies strictly inside the historical segment to test downstream cleaning guards:
1.  **100x Revenue Spikes:** Random multi-million dollar typos representing point-of-sale keying errors.
2.  **Negative Currency Drops:** System glitches logging flat entries of `-$9,500.00`.
3.  **Missing Rows (NaNs):** Simulating a 4% stadium hardware outage resulting in empty revenue rows.

---

##  Phase 2: Automated Data Cleaning & Machine Learning
The core data science pipeline runs inside `phase2_revenue_regressor.py`. It is written to be completely **blind and idempotent**, self-healing any dataset variant using statistical boundaries instead of hardcoded row IDs.

### 1. Type Mitigation & Market-Aware Cleaning
*   **Structural Parsing:** Standardizes mixed text arrays (e.g., matching strings like `"TRUE"`, `"1.0"`, and `"FALSE"`) into machine-learning-ready boolean integers (`1` or `0`).
*   **Market-Segmented IQR Filters:** Rather than applying a global filter that would misclassify small games as typos, the script calculates distinct Interquartile Range (IQR) ceilings and floors for each opponent market tier. Spikes, drops, and `NaN` entries are automatically trapped and scaled to that specific market tier's median revenue.

### 2. Chronological Slicing (No Data Leakage)
Because sports data relies tightly on historical time-series momentum, random shuffling is banned. Slicing isolates a strict **70% Train / 15% Validation / 15% Test partition** entirely from the historical segment (Games 1–243) to guarantee no future timeline information leaks into past model layers.

### 3. Hyperparameter Optimization & Model Selection
The pipeline fits a **Random Forest Regressor** to predict precise continuous expected dollar figures. It runs an automated `GridSearchCV` hyperparameter tuning loop across **144 distinct tree structures** using a `PredefinedSplit` on the validation data.

###  Machine Learning Operational Outputs
When the pipeline executes, it prints a clean diagnostic report to the terminal and freezes the optimal model file as a serialized `revenue_rf_regressor.pkl`:

```text
==================================================
GRIDSEARCHCV TUNING PHASE METRICS
==================================================
Optimal Hyperparameters: {'max_depth': 8, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 50}
Best Validation Score:   \$32,470.20 MAE

==================================================
FINAL UNBIASED TEST REPORT (HISTORICAL ERA)
==================================================
Final Test Mean Absolute Error: \$43,601.25
Final Test R-Squared Score:     0.8782

==================================================
FEATURE IMPORTANCE RANKINGS (BUSINESS INSIGHTS)
==================================================
1. is_home_game            -> 0.7218 (72.2%)
2. opponent_market_encoded -> 0.1453 (14.5%)
3. rain_chance_pct         -> 0.0472 (4.7%)
4. knights_win_pct_10d     -> 0.0314 (3.1%)
5. is_weekend              -> 0.0297 (3.0%)
6. has_promotion           -> 0.0246 (2.5%)
==================================================
```

---

##  Phase 3: Prescriptive Power BI Optimization Console
The reporting layer connects directly to the repository data folder, matching your predictive outputs to your primary schedule using a **One-to-One (`1 <-> 1`) relational data link** on the `game_id` key.

###  Elite Architecture Highlights
*   **Centralized Measures Folder:** Created a hidden placeholder column to compile all active metrics into a unified `_Measures` calculator table pinned to the top of the interface.
*   **Chronological Master Timeline (DAX):** Built a filter-aware timeline chart blending past actual numbers and unplayed machine learning forecasts seamlessly on a single vector using a DAX `COALESCE` statement:
    ```dax
    Unified Revenue Trend = COALESCE(SUM(clean_stadium_data[clean_total_revenue]), SUM(production_predictions[predicted_revenue]))
    ```
*   **Custom Matrix Sorting:** Linked alphabetical market strings (`Large`, `Mid`, `Low`) directly to your numerical `opponent_market_encoded` column, overriding Power BI's default alphabetical sort order.

###  Interactive What-If Simulation Engine
The dashboard acts as a live decision-making suite. By deploying a numeric range parameter slider paired with a visual-level `Promo Action Trigger` filter, users can simulate adding marketing promotions to games and watch the future schedule ratio respond dynamically.

---

##  Model-Driven Promotion Target Dates
By setting the slider to a realistic, cost-aware **1.10 multiplier (10% revenue lift factor)** to respect marginal operational expenses, the dashboard filters out unviable low-market games and surfaces the exact **five calendar target dates** front-office executives must focus on:

*    **Game 244 vs. The Millers** - Weekend home game successfully pushed above moving averages.
*    **Game 245 vs. The Millers** - Weekday home game successfully pushed above moving averages.
*    **Game 247 vs. The Towers** - Weekday game successfully pushed above moving averages.
*    **Game 249 vs. The Towers** - Weekend home game optimized for premium margin lift.
*    **Game 296 vs. The Emeralds** - Late-season weekend matchup targeted for attendance recovery.

By executing this strategic framework, the team's upcoming production performance profile shifts from an unoptimized **19 Above / 23 Below** baseline split to a highly profitable **24 Above / 18 Below ratio**.

---

##  Execution & Replication Instructions
To replicate this entire pipeline locally from your terminal, execute the following steps sequentially:

1.  **Clone the Repository:**
    ```cmd
    git clone https://github.com
    cd sports-business-analytics-lifecycle
    ```
2.  **Generate the Database Warehouse (Phase 1):**
    ```cmd
    python database/database_setup.py
    ```
3.  **Run the Cleaning & Machine Learning Engine (Phase 2):**
    ```cmd
    python database/data_clean_and_analytics.py
    ```
4.  **Launch the Visualization Canvas (Phase 3):**
    Open `dashboards/Stadium_Revenue_Forecast_Optimizer.pbix` in Power BI Desktop and click **Refresh** to instantly stream your updated data tables!

***
For a deep dive into the standalone business justifications, cost controls, and operational strategies of this model, please review the front-office report:
**[View the 1-Page Summary Brief (executive_summary.md)](./executive_summary.md)**
