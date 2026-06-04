# Phase-Level Aviation Fuel Consumption Estimation Using ADS-B Trjectory Data

## An XGBoost Modelling Framework Benchmarked Against a Physics-Based Approach

## Project Overview

This repository contains the code developed as part of the master’s thesis project by Emma Bergsmo and Thea Remman Paulse at the Norwegian University of Science and Technology (NTNU).

The aim of the project is to estimate aircraft fuel consumption using machine learning models based on ADS-B trajectory data, recorded fuel consumption, and engineered features. The results are then compared to AviTeam, a physics-based fuel estimation model. The project follows a complete data pipeline, from data retrieval and preprocessing to feature generation, model training, evaluation, and visualization of results.

The is based on ADS-B data from the OpenSky Network and fuel records from the Norwegian Air Shuttle ASA. XGBoost models are trained and evaluated for full-flight fuel consumption as well as for individual flight phases: takeoff, climb, cruise, descent, and landing.

## Data

- Opensky Network Trino Database
  - `flights_data4`
  - `state_vectors_data4`
- Norwegian Air Shuttle ASA recorded fuel consumption in 2022
- Norwegian Meteorological Institute Frost API

## Data Privacy

Since private recorded fuel data from Norwegian Air Shuttle ASA is used, the input data is excluded from Git.

## Repository Structure

- `src/retrieve_data/`: scripts for retrieving data from OpenSky and Frost API
- `src/preprocessing/`: scripts for matching flights, assigning flight IDs, and identifying flight phases
- `src/create_features/`: scripts for creating model-ready feature tables, including physics, weather, heading, and time features
- `src/aviteam_ready/`: scripts for preparing and formatting data to be compatible with AviTeam
- `notebooks/`: XGBoost models for different flight phases and feature sets
- `plotting/`: scripts used to generate thesis plots and figures
- `outputs/`: generated figures, models, predictions, and tables

```text
Fuel_estimation_machine_learning/
├── README.md
├── requirements.txt
├── setup.sh
├── .gitignore
├── norwegian_data.csv  # Not tracked in Git
├── data.sqlite   # Not tracked in Git
│
├── src/
│ ├── data_retreival/
│ │ ├── trino_client.py
│ │ ├── retrieve_adsb_data.py
│ │ ├── retrieve_norwegian.py
│ │ ├── retrieve_norwegian_domestic.py
│ │ └── retrieve_weather_data.py
│ │
│ ├── preprocessing/
│ │ ├── set_flight_id.py
│ │ ├── match_flights.py
│ │ └── set_flight_phases.py
│ │
│ ├── feature_engineering/
│ │ ├── create_trajectory_features.py
│ │ ├── create_physics_features.py
│ │ ├── create_heading_features.py
│ │ ├── create_time_features.py
│ │ └── create_weather_features.py
│ │
│ └── aviteam_export/
│   ├── aviteam_ready.py
│   ├── sqlite_to_csv.py
│   └── sqlite_to_h5.py
│
├── notebooks/
│ ├── xgboost/
│ │ ├── total_pred.ipynb
│ │ ├── takeoff.ipynb
│ │ ├── climb.ipynb
│ │ ├── cruise.ipynb
│ │ ├── descent.ipynb
│ │ └── landing.ipynb
│ │
│ ├── xgboost_physics/
│ │ ├── total_pred_physics.ipynb
│ │ ├── takeoff_physics.ipynb
│ │ ├── climb_physics.ipynb
│ │ ├── cruise_physics.ipynb
│ │ ├── descent_physics.ipynb
│ │ └── landing_physics.ipynb
│ │
│ └── xgboost_weather_time/
│   ├── total_pred_weather_time.ipynb
│   ├── takeoff_weather_time.ipynb
│   ├── climb_weather_time.ipynb
│   ├── cruise_weather_time.ipynb
│   ├── descent_weather_time.ipynb
│   └── landing_weather_time.ipynb
│
├── plotting/
│ ├── normal_distribution.py
│ └── normal_distribution_delta.py
│ └── route_level_errors.py
│ └── routes_map.py
│ └── weather_stations_map.py
│ └── plot_style.py
│ └── feature_importance_panels.py
│ └── phase_regression_panels.py
│
└── outputs/    # Not tracked in Git
```

## Workflow

Machine learning:

1. Retrieve domestic Norwegian Air Shuttle ASA flights from OpenSky
2. Match OpenSky data with Norwegian Air Shuttle ASA dataset
3. Retrieve ADS-B trajectory data for matched flights
4. Create unique flight identifier
5. Assign flight phases
6. Create features from ADS-B dataset
7. Train XGBoost models in the notebooks
8. Generate plots and output files

AviTeam:

1. Retrieve domestic Norwegian Air Shuttle ASA flights from OpenSky
2. Match OpenSky data with Norwegian Air Shuttle ASA dataset
3. Retrieve ADS-B trajectory data for matched flights
4. Create unique flight identifier
5. Assign flight phases
6. Make ADS-B dataset compatible with AviTeam
7. Save and export dataset as HDF5

## Outputs

- Data tables written to data.sqlite
- Feature tables written data.sqlite
- XGBoost model performnace presented inside notebooks
- Plotting results saved in outputs folder

## Setup

```text
chmod +x setup.sh
./setup.sh
```

Must create an .env file to connect to Trino database, which includes:

```text
OPENSKY_USER={username}
```

## How to Run Files

```text
python src/retrieve_data/retrieve_norwegian_domestic.py
```
