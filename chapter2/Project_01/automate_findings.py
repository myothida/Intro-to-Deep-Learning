import pandas as pd
import numpy as np
import os
import time
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ModelTrainer:
    def __init__(self, data_path='./data/SGHDB2017-2024_clean.csv'):
        """Initialize the trainer with data loading and preprocessing"""
        self.data_path = data_path
        self.load_and_prepare_data()
        
    def load_and_prepare_data(self):
        """Load and preprocess the data"""
        print("Loading and preparing data...")
        
        # Load data
        df = pd.read_csv(self.data_path)
        self.y = df['adjusted_price'].values
        self.X = df.drop(columns='adjusted_price')
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42
        )
        
        # Prepare features for scaling
        continuous_columns = ['flat_type', 'floor_area_sqm', 'floor', 'remaining_lease_months']
        binary_columns = df.columns.difference(continuous_columns + ['adjusted_price']).tolist()
        
        # Scale continuous features
        scaler = StandardScaler()
        X_train_continuous = scaler.fit_transform(X_train[continuous_columns])
        X_test_continuous = scaler.transform(X_test[continuous_columns])
        
        # Combine scaled continuous and binary features
        self.X_train = np.hstack([X_train_continuous, X_train[binary_columns].values])
        self.X_test = np.hstack([X_test_continuous, X_test[binary_columns].values])
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"Data prepared: Train shape {self.X_train.shape}, Test shape {self.X_test.shape}")
    
    def create_model(self, layers):
        """Create a neural network model with specified layer configuration"""
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(shape=[self.X_train.shape[1]]))
        
        # Add hidden layers
        for neurons in layers[:-1]:
            model.add(tf.keras.layers.Dense(neurons, activation='relu'))
        
        # Add output layer
        model.add(tf.keras.layers.Dense(layers[-1]))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def save_model_summary(self, model, save_path):
        """Save model summary to text file"""
        with open(os.path.join(save_path, 'model_summary.txt'), 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
            f.write(f"\nTotal Parameters: {model.count_params():,}\n")
    
    def plot_learning_curves(self, history, save_path):
        """Plot and save learning curves"""
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
        plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(history.history['loss'], label='Training Loss (Log Scale)', linewidth=2)
        plt.plot(history.history['val_loss'], label='Validation Loss (Log Scale)', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Log Scale)')
        plt.title('Training and Validation Loss Over Epochs (Log Scale)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'learning_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_predictions(self, y_actual, y_predicted, save_path, dataset_name):
        """Plot and save prediction scatter plot"""
        plt.figure(figsize=(10, 8))
        
        plt.scatter(y_actual, y_predicted, alpha=0.6, s=20)
        
        # Perfect prediction line
        min_val = min(y_actual.min(), y_predicted.min())
        max_val = max(y_actual.max(), y_predicted.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Predicted vs Actual Values - {dataset_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add R² score to plot
        r2 = r2_score(y_actual, y_predicted)
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'predictions_{dataset_name.lower()}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def evaluate_model(self, model, save_path, training_time):
        """Evaluate model and save results"""
        # Predictions
        y_train_pred = model.predict(self.X_train, verbose=0)
        y_test_pred = model.predict(self.X_test, verbose=0)
        
        # Calculate metrics
        results = pd.DataFrame(columns=['Train', 'Test'])
        
        # Train metrics
        results.loc['Root Mean Squared Error', 'Train'] = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        results.loc['Mean Absolute Error', 'Train'] = mean_absolute_error(self.y_train, y_train_pred)
        results.loc['Mean Absolute Percentage Error', 'Train'] = mean_absolute_percentage_error(self.y_train, y_train_pred) * 100
        results.loc['R2 Score', 'Train'] = r2_score(self.y_train, y_train_pred)
        
        # Test metrics
        results.loc['Root Mean Squared Error', 'Test'] = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        results.loc['Mean Absolute Error', 'Test'] = mean_absolute_error(self.y_test, y_test_pred)
        results.loc['Mean Absolute Percentage Error', 'Test'] = mean_absolute_percentage_error(self.y_test, y_test_pred) * 100
        results.loc['R2 Score', 'Test'] = r2_score(self.y_test, y_test_pred)
        
        # Add training time
        results.loc['Training Time (seconds)', 'Train'] = training_time
        results.loc['Training Time (seconds)', 'Test'] = training_time
        
        # Round results
        results = results.astype('float64').round(4)
        
        # Save results
        results.to_csv(os.path.join(save_path, 'evaluation_results.csv'))
        
        # Create prediction plots
        self.plot_predictions(self.y_train, y_train_pred.flatten(), save_path, 'Train')
        self.plot_predictions(self.y_test, y_test_pred.flatten(), save_path, 'Test')
        
        return results
    
    def train_single_model(self, model_config, epochs, model_name, epoch_name):
        """Train a single model configuration"""
        print(f"\nTraining {model_name} - {epoch_name}")
        
        # Create directory
        save_dir = os.path.join('results', f'{model_name}_{epoch_name}')
        os.makedirs(save_dir, exist_ok=True)
        
        # Create model
        model = self.create_model(model_config)
        
        # Save model summary
        self.save_model_summary(model, save_dir)
        
        # Train model
        start_time = time.time()
        history = model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(self.X_test, self.y_test),
            verbose=1
        )
        training_time = time.time() - start_time
        
        # Save learning curves
        self.plot_learning_curves(history, save_dir)
        
        # Evaluate and save results
        results = self.evaluate_model(model, save_dir, training_time)
        
        # Save training history
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(os.path.join(save_dir, 'training_history.csv'), index=False)
        
        print(f"Completed {model_name} - {epoch_name}")
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Test R² Score: {results.loc['R2 Score', 'Test']:.4f}")
        
        return results
    
    def run_all_experiments(self):
        """Run all model architectures with all epoch configurations"""
        # Model configurations
        model_configs = {
            'model1': [32, 16, 1],
            'model2': [32, 16, 8, 1],
            'model3': [64, 32, 16, 8, 1],
            'model4': [128, 64, 32, 16, 8, 1],
            'model5': [256, 128, 64, 32, 16, 8, 1]
        }
        
        # Epoch configurations
        epoch_configs = [10, 20, 30, 40, 50]
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
        
        # Store all results for summary
        all_results = []
        
        # Run experiments
        for model_name, model_config in model_configs.items():
            print(f"\n{'='*60}")
            print(f"Starting experiments for {model_name}: {' -> '.join(map(str, model_config))}")
            print(f"{'='*60}")
            
            for epochs in epoch_configs:
                epoch_name = f'epoch_{epochs}'
                
                try:
                    results = self.train_single_model(
                        model_config, epochs, model_name, epoch_name
                    )
                    
                    # Store summary info
                    summary_info = {
                        'Model': model_name,
                        'Architecture': ' -> '.join(map(str, model_config)),
                        'Epochs': epochs,
                        'Train_R2': results.loc['R2 Score', 'Train'],
                        'Test_R2': results.loc['R2 Score', 'Test'],
                        'Train_RMSE': results.loc['Root Mean Squared Error', 'Train'],
                        'Test_RMSE': results.loc['Root Mean Squared Error', 'Test'],
                        'Training_Time': results.loc['Training Time (seconds)', 'Train']
                    }
                    all_results.append(summary_info)
                    
                except Exception as e:
                    print(f"Error training {model_name} - {epoch_name}: {str(e)}")
                    continue
        
        # Save summary results
        summary_df = pd.DataFrame(all_results)
        summary_df.to_csv('results/experiment_summary.csv', index=False)
        
        print(f"\n{'='*60}")
        print("ALL EXPERIMENTS COMPLETED!")
        print(f"{'='*60}")
        print(f"Results saved in 'results' directory")
        print(f"Summary saved as 'results/experiment_summary.csv'")
        
        return summary_df

def main():
    """Main function to run all experiments"""
    print("Starting Automated Model Training Pipeline")
    print("="*60)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Run all experiments
    summary = trainer.run_all_experiments()
    
    # Display summary
    print("\nExperiment Summary:")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()