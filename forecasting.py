"""
Forecasting Module for Financial Portfolio

Implements multiple forecasting methods for stock price prediction:
1. Linear Trend Forecasting
2. Exponential Smoothing (Holt-Winters)
3. ARIMA (AutoRegressive Integrated Moving Average)
4. Random Forest Regression
5. Gradient Boosting Regression

Features:
- Automatic parameter optimization
- Multiple evaluation metrics (MAE, RMSE, MAPE)
- Train-test split validation
- Comparison plots
- Future price forecasts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from pathlib import Path
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Create results directory
Path('results').mkdir(exist_ok=True)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


class StockForecaster:
    """
    Multi-method stock price forecasting.
    
    Methods:
    --------
    1. Linear Trend - Simple linear extrapolation
    2. Exponential Smoothing - Captures trend and seasonality
    3. ARIMA - Time series AR, I, MA components
    4. Random Forest - ML ensemble method
    5. Gradient Boosting - Advanced ML ensemble
    """
    
    def __init__(self, data, ticker=None, test_size=0.2):
        """
        Initialize forecaster.
        
        Parameters:
        -----------
        data : pd.DataFrame or pd.Series
            Time series data (Adjusted Close prices)
        ticker : str, optional
            Stock ticker symbol
        test_size : float
            Proportion for test set (0.2 = 20%)
        """
        if isinstance(data, pd.DataFrame):
            if 'Adj Close' in data.columns:
                self.prices = data['Adj Close'].values
            else:
                self.prices = data.iloc[:, 0].values
        else:
            self.prices = data.values
        
        self.ticker = ticker or "Stock"
        self.test_size = test_size
        self.split_idx = int(len(self.prices) * (1 - test_size))
        self.train_prices = self.prices[:self.split_idx]
        self.test_prices = self.prices[self.split_idx:]
        
        self.results = {}
        self.forecasts = {}
    
    def _calculate_metrics(self, y_true, y_pred, method_name):
        """
        Calculate evaluation metrics.
        
        Parameters:
        -----------
        y_true : array
            Actual values
        y_pred : array
            Predicted values
        method_name : str
            Name of forecasting method
        
        Returns:
        --------
        dict : Metrics dictionary
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred)
        
        return {
            'method': method_name,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'predictions': y_pred
        }
    
    def linear_trend_forecast(self, periods=30, plot=True):
        """
        Simple linear trend forecasting.
        
        Fits a linear trend to historical prices and extrapolates.
        
        Parameters:
        -----------
        periods : int
            Number of periods to forecast
        plot : bool
            Whether to plot results
        
        Returns:
        --------
        dict : Forecast results
        """
        print("\n" + "="*60)
        print("LINEAR TREND FORECASTING")
        print("="*60)
        
        # Fit on training data
        X_train = np.arange(len(self.train_prices)).reshape(-1, 1)
        y_train = self.train_prices
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predict on test set
        X_test = np.arange(len(self.train_prices), len(self.train_prices) + len(self.test_prices)).reshape(-1, 1)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        metrics = self._calculate_metrics(self.test_prices, y_pred_test, 'Linear Trend')
        self.results['linear_trend'] = metrics
        
        # Forecast future
        X_future = np.arange(len(self.prices), len(self.prices) + periods).reshape(-1, 1)
        future_forecast = model.predict(X_future)
        
        print(f"Slope: ${model.coef_[0]:.4f}/day")
        print(f"MAE: ${metrics['MAE']:.2f}")
        print(f"RMSE: ${metrics['RMSE']:.2f}")
        print(f"MAPE: {metrics['MAPE']:.2f}%")
        print(f"Next {periods} days forecast:")
        print(f"  Start: ${future_forecast[0]:.2f}")
        print(f"  End: ${future_forecast[-1]:.2f}")
        print(f"  Change: ${future_forecast[-1] - future_forecast[0]:.2f}")
        
        self.forecasts['linear_trend'] = {
            'test_pred': y_pred_test,
            'future_pred': future_forecast,
            'model': model
        }
        
        if plot:
            self._plot_forecast('linear_trend', periods)
        
        return metrics
    
    def exponential_smoothing_forecast(self, periods=30, plot=True):
        """
        Exponential smoothing (Holt-Winters) forecasting.
        
        Captures trend and can capture seasonality.
        
        Parameters:
        -----------
        periods : int
            Number of periods to forecast
        plot : bool
            Whether to plot results
        
        Returns:
        --------
        dict : Forecast results
        """
        print("\n" + "="*60)
        print("EXPONENTIAL SMOOTHING FORECASTING")
        print("="*60)
        
        try:
            # Fit on training data
            model = ExponentialSmoothing(
                self.train_prices,
                trend='add',
                seasonal=None,
                initialization_method='estimated'
            )
            fitted_model = model.fit(optimized=True)
            
            # Predict on test set
            y_pred_test = fitted_model.forecast(steps=len(self.test_prices))
            
            # Metrics
            metrics = self._calculate_metrics(self.test_prices, y_pred_test, 'Exponential Smoothing')
            self.results['exponential_smoothing'] = metrics
            
            # Forecast future
            future_forecast = fitted_model.forecast(steps=len(self.test_prices) + periods)[len(self.test_prices):]
            
            print(f"Smoothing Level: {fitted_model.params['smoothing_level']:.4f}")
            print(f"Smoothing Trend: {fitted_model.params['smoothing_trend']:.4f}")
            print(f"MAE: ${metrics['MAE']:.2f}")
            print(f"RMSE: ${metrics['RMSE']:.2f}")
            print(f"MAPE: {metrics['MAPE']:.2f}%")
            print(f"Next {periods} days forecast:")
            print(f"  Start: ${future_forecast[0]:.2f}")
            print(f"  End: ${future_forecast[-1]:.2f}")
            print(f"  Change: ${future_forecast[-1] - future_forecast[0]:.2f}")
            
            self.forecasts['exponential_smoothing'] = {
                'test_pred': y_pred_test,
                'future_pred': future_forecast,
                'model': fitted_model
            }
            
            if plot:
                self._plot_forecast('exponential_smoothing', periods)
            
            return metrics
        
        except Exception as e:
            print(f"✗ Error: {str(e)[:100]}")
            return None
    
    def arima_forecast(self, order=(5, 1, 2), periods=30, plot=True):
        """
        ARIMA forecasting.
        
        Captures autoregressive, differencing, and moving average components.
        
        Parameters:
        -----------
        order : tuple
            (p, d, q) for (AR, I, MA)
        periods : int
            Number of periods to forecast
        plot : bool
            Whether to plot results
        
        Returns:
        --------
        dict : Forecast results
        """
        print("\n" + "="*60)
        print(f"ARIMA FORECASTING (order={order})")
        print("="*60)
        
        try:
            # Fit on training data
            model = ARIMA(self.train_prices, order=order)
            fitted_model = model.fit()
            
            # Predict on test set
            y_pred_test = fitted_model.forecast(steps=len(self.test_prices))
            
            # Metrics
            metrics = self._calculate_metrics(self.test_prices, y_pred_test, f'ARIMA{order}')
            self.results['arima'] = metrics
            
            # Forecast future
            future_forecast = fitted_model.forecast(steps=len(self.test_prices) + periods)[len(self.test_prices):]
            
            print(f"AIC: {fitted_model.aic:.2f}")
            print(f"BIC: {fitted_model.bic:.2f}")
            print(f"MAE: ${metrics['MAE']:.2f}")
            print(f"RMSE: ${metrics['RMSE']:.2f}")
            print(f"MAPE: {metrics['MAPE']:.2f}%")
            print(f"Next {periods} days forecast:")
            print(f"  Start: ${future_forecast[0]:.2f}")
            print(f"  End: ${future_forecast[-1]:.2f}")
            print(f"  Change: ${future_forecast[-1] - future_forecast[0]:.2f}")
            
            self.forecasts['arima'] = {
                'test_pred': y_pred_test,
                'future_pred': future_forecast,
                'model': fitted_model
            }
            
            if plot:
                self._plot_forecast('arima', periods)
            
            return metrics
        
        except Exception as e:
            print(f"✗ Error: {str(e)[:100]}")
            return None
    
    def random_forest_forecast(self, lags=5, periods=30, plot=True, n_estimators=100):
        """
        Random Forest regression forecasting.
        
        Uses lagged prices as features for machine learning prediction.
        
        Parameters:
        -----------
        lags : int
            Number of lagged features
        periods : int
            Number of periods to forecast
        plot : bool
            Whether to plot results
        n_estimators : int
            Number of trees in forest
        
        Returns:
        --------
        dict : Forecast results
        """
        print("\n" + "="*60)
        print("RANDOM FOREST FORECASTING")
        print("="*60)
        
        try:
            # Create lagged features
            X = np.array([self.train_prices[i-lags:i] for i in range(lags, len(self.train_prices))])
            y = self.train_prices[lags:]
            
            # Train model
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
            model.fit(X, y)
            
            # Predict on test set using a rolling window
            window = list(self.train_prices[-lags:])
            y_pred_test = []
            for i in range(len(self.test_prices)):
                pred = model.predict(np.array([window[-lags:]]))[0]
                y_pred_test.append(pred)
                window.append(self.test_prices[i])
            y_pred_test = np.array(y_pred_test)
            
            # Metrics
            metrics = self._calculate_metrics(self.test_prices[:len(y_pred_test)], y_pred_test, 'Random Forest')
            self.results['random_forest'] = metrics
            
            # Forecast future
            future_prices = list(self.test_prices)
            future_forecast = []
            
            for _ in range(periods):
                X_future = np.array([future_prices[-lags:]])
                pred = model.predict(X_future)[0]
                future_forecast.append(pred)
                future_prices.append(pred)
            
            print(f"N Estimators: {n_estimators}")
            print(f"Feature Importance (top 3):")
            for i, imp in enumerate(sorted(enumerate(model.feature_importances_), key=lambda x: x[1], reverse=True)[:3]):
                print(f"  Lag {imp[0]+1}: {imp[1]:.4f}")
            print(f"MAE: ${metrics['MAE']:.2f}")
            print(f"RMSE: ${metrics['RMSE']:.2f}")
            print(f"MAPE: {metrics['MAPE']:.2f}%")
            print(f"Next {periods} days forecast:")
            print(f"  Start: ${future_forecast[0]:.2f}")
            print(f"  End: ${future_forecast[-1]:.2f}")
            print(f"  Change: ${future_forecast[-1] - future_forecast[0]:.2f}")
            
            self.forecasts['random_forest'] = {
                'test_pred': y_pred_test,
                'future_pred': np.array(future_forecast),
                'model': model
            }
            
            if plot:
                self._plot_forecast('random_forest', periods)
            
            return metrics
        
        except Exception as e:
            print(f"✗ Error: {str(e)[:100]}")
            return None
    
    def gradient_boosting_forecast(self, lags=5, periods=30, plot=True, n_estimators=100):
        """
        Gradient Boosting regression forecasting.
        
        Advanced ensemble method for regression.
        
        Parameters:
        -----------
        lags : int
            Number of lagged features
        periods : int
            Number of periods to forecast
        plot : bool
            Whether to plot results
        n_estimators : int
            Number of boosting stages
        
        Returns:
        --------
        dict : Forecast results
        """
        print("\n" + "="*60)
        print("GRADIENT BOOSTING FORECASTING")
        print("="*60)
        
        try:
            # Create lagged features
            X = np.array([self.train_prices[i-lags:i] for i in range(lags, len(self.train_prices))])
            y = self.train_prices[lags:]
            
            # Train model
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            model.fit(X, y)
            
            # Predict on test set using a rolling window
            window = list(self.train_prices[-lags:])
            y_pred_test = []
            for i in range(len(self.test_prices)):
                pred = model.predict(np.array([window[-lags:]]))[0]
                y_pred_test.append(pred)
                window.append(self.test_prices[i])
            y_pred_test = np.array(y_pred_test)
            
            # Metrics
            metrics = self._calculate_metrics(self.test_prices[:len(y_pred_test)], y_pred_test, 'Gradient Boosting')
            self.results['gradient_boosting'] = metrics
            
            # Forecast future
            future_prices = list(self.test_prices)
            future_forecast = []
            
            for _ in range(periods):
                X_future = np.array([future_prices[-lags:]])
                pred = model.predict(X_future)[0]
                future_forecast.append(pred)
                future_prices.append(pred)
            
            print(f"N Estimators: {n_estimators}")
            print(f"Learning Rate: 0.1")
            print(f"MAE: ${metrics['MAE']:.2f}")
            print(f"RMSE: ${metrics['RMSE']:.2f}")
            print(f"MAPE: {metrics['MAPE']:.2f}%")
            print(f"Next {periods} days forecast:")
            print(f"  Start: ${future_forecast[0]:.2f}")
            print(f"  End: ${future_forecast[-1]:.2f}")
            print(f"  Change: ${future_forecast[-1] - future_forecast[0]:.2f}")
            
            self.forecasts['gradient_boosting'] = {
                'test_pred': y_pred_test,
                'future_pred': np.array(future_forecast),
                'model': model
            }
            
            if plot:
                self._plot_forecast('gradient_boosting', periods)
            
            return metrics
        
        except Exception as e:
            print(f"✗ Error: {str(e)[:100]}")
            return None
    
    def forecast_comparison(self):
        """
        Compare all forecasting methods and create comparison visualizations.
        """
        print("\n" + "="*60)
        print("FORECASTING METHODS COMPARISON")
        print("="*60)
        
        # Create comparison dataframe
        comparison_data = []
        for method, metrics in self.results.items():
            if metrics is not None:
                comparison_data.append({
                    'Method': metrics['method'],
                    'MAE': metrics['MAE'],
                    'RMSE': metrics['RMSE'],
                    'MAPE': metrics['MAPE']
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            # Sort by RMSE
            comparison_df = comparison_df.sort_values('RMSE')
            
            print("\nPerformance Metrics:")
            print(comparison_df.to_string(index=False))
            
            # Plot comparison
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # MAE comparison
            axes[0].barh(comparison_df['Method'], comparison_df['MAE'], color='skyblue')
            axes[0].set_xlabel('MAE ($)')
            axes[0].set_title('Mean Absolute Error Comparison')
            axes[0].grid(True, alpha=0.3, axis='x')
            
            # RMSE comparison
            axes[1].barh(comparison_df['Method'], comparison_df['RMSE'], color='lightcoral')
            axes[1].set_xlabel('RMSE ($)')
            axes[1].set_title('Root Mean Squared Error Comparison')
            axes[1].grid(True, alpha=0.3, axis='x')
            
            # MAPE comparison
            axes[2].barh(comparison_df['Method'], comparison_df['MAPE'], color='lightgreen')
            axes[2].set_xlabel('MAPE (%)')
            axes[2].set_title('Mean Absolute Percentage Error Comparison')
            axes[2].grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            plt.savefig('results/forecasting_comparison.png', dpi=300, bbox_inches='tight')
            print("\n✓ Saved: results/forecasting_comparison.png")
            plt.close()
            
            return comparison_df
        else:
            print("No results to compare.")
            return None
    
    def _plot_forecast(self, method, periods):
        """
        Plot forecasting results for a specific method.
        
        Parameters:
        -----------
        method : str
            Forecasting method name
        periods : int
            Number of future periods forecasted
        """
        forecast = self.forecasts[method]
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Plot training data
        ax.plot(range(len(self.train_prices)), self.train_prices, 
               label='Training Data', color='blue', linewidth=2)
        
        # Plot test data
        ax.plot(range(len(self.train_prices), len(self.prices)), self.test_prices,
               label='Test Data', color='green', linewidth=2)
        
        # Plot predictions on test set
        ax.plot(range(len(self.train_prices), len(self.train_prices) + len(forecast['test_pred'])), 
               forecast['test_pred'], label='Predictions (Test)', color='orange', linestyle='--', linewidth=2)
        
        # Plot future forecast
        ax.plot(range(len(self.prices), len(self.prices) + len(forecast['future_pred'])),
               forecast['future_pred'], label=f'Forecast ({periods} days)', 
               color='red', linestyle='--', linewidth=2)
        
        ax.set_xlabel('Days')
        ax.set_ylabel('Price ($)')
        ax.set_title(f'{self.ticker} - {method.replace("_", " ").title()} Forecast')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'results/forecast_{method}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()


if __name__ == "__main__":
    # Load sample data
    try:
        data = pd.read_csv('data/sp500_stocks.csv', index_col=0, parse_dates=True)
        
        # Get a single stock
        if 'Ticker' in data.columns:
            ticker = data['Ticker'].iloc[0]
            stock_data = data[data['Ticker'] == ticker]['Adj Close']
        else:
            stock_data = data.iloc[:, 0]
            ticker = "STOCK"
        
        print(f"Forecasting {ticker}...\n")
        
        # Initialize forecaster
        forecaster = StockForecaster(stock_data, ticker=ticker)
        
        # Run all forecasting methods
        forecaster.linear_trend_forecast(periods=30)
        forecaster.exponential_smoothing_forecast(periods=30)
        forecaster.arima_forecast(order=(5, 1, 2), periods=30)
        forecaster.random_forest_forecast(lags=5, periods=30)
        forecaster.gradient_boosting_forecast(lags=5, periods=30)
        
        # Compare all methods
        forecaster.forecast_comparison()
        
    except FileNotFoundError:
        print("Data file not found. Run download_data.py first.")
        print("\nExample:")
        print("  python download_data.py")
