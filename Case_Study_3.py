# energy_analysis_gradient_boosting.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class BuildingEnergyAnalyzer:
    def __init__(self):
        self.model = GradientBoostingRegressor(n_estimators=200, 
                                             learning_rate=0.1,
                                             max_depth=5,
                                             random_state=42)
        self.scaler = StandardScaler()
        self.feature_importance = None
        
    def load_and_prepare_data(self, filename='building_energy_data.csv'):
        # Load data
        data = pd.read_csv(filename)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        return data

    def train_model(self, data):
        # Feature selection
        features = ['occupancy', 'outdoor_temp', 'humidity', 'wind_speed', 
                   'solar_radiation', 'hour', 'day_of_week']
        X = data[features]
        y = data['energy_consumption']
        
        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance - MSE: {mse:.2f}, R2 Score: {r2:.2f}")
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': features,
            'importance': self.model.feature_importances_
        })
        
        return X_train_scaled, X_test_scaled, y_train, y_test

    def identify_inefficiencies(self, data):
        features = ['occupancy', 'outdoor_temp', 'humidity', 'wind_speed', 
                   'solar_radiation', 'hour', 'day_of_week']
        X_scaled = self.scaler.transform(data[features])
        predicted_consumption = self.model.predict(X_scaled)
        
        data['predicted_consumption'] = predicted_consumption
        data['consumption_difference'] = data['energy_consumption'] - data['predicted_consumption']
        
        inefficiencies = data[data['consumption_difference'] > data['consumption_difference'].quantile(0.95)]
        return inefficiencies

    def generate_recommendations(self, inefficiencies):
        recommendations = []
        
        for _, row in inefficiencies.iterrows():
            if row['occupancy'] > 400:
                recommendations.append(f"High occupancy ({row['occupancy']}) at {row['timestamp']}: "
                                     "Optimize HVAC zoning")
            if row['outdoor_temp'] > 28:
                recommendations.append(f"High temp ({row['outdoor_temp']}°C) at {row['timestamp']}: "
                                     "Increase cooling efficiency")
            if row['humidity'] > 70:
                recommendations.append(f"High humidity ({row['humidity']}%) at {row['timestamp']}: "
                                     "Enhance dehumidification")
            if row['solar_radiation'] > 300 and row['hour'] in range(10, 16):
                recommendations.append(f"High solar radiation ({row['solar_radiation']} W/m²) at {row['timestamp']}: "
                                     "Adjust window shading")
                
        return recommendations

def main():
    analyzer = BuildingEnergyAnalyzer()
    
    print("Loading data...")
    building_data = analyzer.load_and_prepare_data()
    
    print("\nTraining model...")
    X_train_scaled, X_test_scaled, y_train, y_test = analyzer.train_model(building_data)
    
    print("\nIdentifying energy inefficiencies...")
    inefficiencies = analyzer.identify_inefficiencies(building_data)
    print(f"Found {len(inefficiencies)} periods of potential inefficiency")
    
    print("\nGenerating optimization recommendations...")
    recommendations = analyzer.generate_recommendations(inefficiencies)
    for i, rec in enumerate(recommendations[:5], 1):
        print(f"{i}. {rec}")
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=analyzer.feature_importance)
    plt.title('Feature Importance in Energy Consumption Prediction')
    plt.show()

if __name__ == "__main__":
    main()

# Additional features for model improvement:
# 1. Indoor temperature readings
# 2. Building age and insulation properties
# 3. HVAC system efficiency ratings
# 4. Occupancy patterns by room/zone
# 5. Energy price data

# Steps for building managers:
# 1. Adjust temperature setpoints based on predicted demand
# 2. Schedule equipment operation during optimal weather conditions
# 3. Implement occupancy-based lighting controls
# 4. Plan retrofits based on inefficiency patterns
# 5. Optimize ventilation timing

# Net-zero contributions:
# 1. Minimizes energy waste through precise predictions
# 2. Supports solar energy integration with radiation data
# 3. Enables peak load shifting
# 4. Optimizes existing systems without major retrofits
# 5. Provides data for long-term efficiency planning