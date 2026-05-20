"""
Regression Analysis Module for Financial Portfolio

Performs linear regression, log transformations, and statistical analysis
on stock price data to identify trends and relationships.

Features:
- Linear regression analysis
- Log10 and natural log transformations
- Multiple regression with multiple features
- Comprehensive statistical diagnostics
- Publication-quality visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Create results directory
Path('results').mkdir(exist_ok=True)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


class RegressionAnalyzer:
    """
    Comprehensive regression analysis for financial data.
    
    Supports:
    - Linear regression analysis
    - Log transformation (log10, natural log)
    - Multiple regression
    - Statistical metrics and diagnostics
    """
    
    def __init__(self, data, ticker=None):
        """
        Initialize the regression analyzer.
        
        Parameters:
        -----------
        data : pd.DataFrame or pd.Series
            Financial time series data
        ticker : str, optional
            Stock ticker symbol for reference
        """
        self.data = data.copy()
        self.ticker = ticker or "Stock"
        self.results = {}
        self.log_data = {}
        
    def prepare_regression_data(self, target_col='Adj Close', lag=1):
        """
        Prepare data for regression analysis.
        
        Parameters:
        -----------
        target_col : str
            Column to use as target variable
        lag : int
            Number of days to lag for time-series regression
        
        Returns:
        --------
        pd.DataFrame : Prepared regression data
        """
        if target_col not in self.data.columns:
            raise ValueError(f"Column {target_col} not found in data")
        
        df = self.data[[target_col]].copy()
        df['Days'] = np.arange(len(df))
        df['Price_Lag1'] = df[target_col].shift(lag)
        df['Return'] = df[target_col].pct_change() * 100
        df['Log_Price'] = np.log10(df[target_col])
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def linear_regression(self, target_col='Adj Close', plot=True):
        """
        Perform simple linear regression on price vs time.
        
        Parameters:
        -----------
        target_col : str
            Column to analyze
        plot : bool
            Whether to plot results
        
        Returns:
        --------
        dict : Regression statistics and model
        """
        df = self.prepare_regression_data(target_col)
        
        X = df[['Days']].values
        y = df[target_col].values
        
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Calculate metrics
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        slope = model.coef_[0]
        intercept = model.intercept_
        
        results = {
            'model': model,
            'slope': slope,
            'intercept': intercept,
            'r2_score': r2,
            'rmse': rmse,
            'mae': mae,
            'y_pred': y_pred,
            'X': X.flatten(),
            'y': y,
            'daily_change': slope
        }
        
        self.results['linear_regression'] = results
        
        if plot:
            self._plot_linear_regression(results, target_col)
        
        print(f"\n{'='*60}")
        print(f"LINEAR REGRESSION ANALYSIS - {self.ticker}")
        print(f"{'='*60}")
        print(f"Slope (Daily Change): ${slope:.6f}/day")
        print(f"Intercept: ${intercept:.2f}")
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: ${rmse:.2f}")
        print(f"MAE: ${mae:.2f}")
        print(f"Annual Change Rate: ${slope * 252:.2f}/year")
        
        return results
    
    def log_regression(self, target_col='Adj Close', base=10, plot=True):
        """
        Perform regression on log-transformed prices.
        
        Log transformation helps with:
        - Stabilizing variance
        - Handling percentage changes
        - Identifying exponential growth patterns
        
        Parameters:
        -----------
        target_col : str
            Column to analyze
        base : int
            Logarithm base (10 or np.e for natural log)
        plot : bool
            Whether to plot results
        
        Returns:
        --------
        dict : Log regression statistics
        """
        df = self.prepare_regression_data(target_col)
        
        if base == 10:
            y_log = np.log10(df[target_col].values)
            log_type = "Log10"
        else:
            y_log = np.log(df[target_col].values)
            log_type = "Natural Log (ln)"
        
        X = df[['Days']].values
        
        model = LinearRegression()
        model.fit(X, y_log)
        y_log_pred = model.predict(X)
        
        # Convert back to original scale for interpretation
        if base == 10:
            y_original_scale = np.power(10, y_log_pred)
        else:
            y_original_scale = np.exp(y_log_pred)
        
        r2 = r2_score(y_log, y_log_pred)
        rmse = np.sqrt(mean_squared_error(y_log, y_log_pred))
        
        # Calculate percentage change per day
        pct_change_per_day = (np.power(10, model.coef_[0]) - 1) * 100 if base == 10 else (np.exp(model.coef_[0]) - 1) * 100
        
        results = {
            'model': model,
            'y_log': y_log,
            'y_log_pred': y_log_pred,
            'y_original_scale': y_original_scale,
            'X': X.flatten(),
            'y': df[target_col].values,
            'r2_score': r2,
            'rmse': rmse,
            'slope_log': model.coef_[0],
            'intercept_log': model.intercept_,
            'pct_change_per_day': pct_change_per_day,
            'base': base,
            'log_type': log_type
        }
        
        self.results['log_regression'] = results
        self.log_data['log_prices'] = y_log
        
        if plot:
            self._plot_log_regression(results, target_col)
        
        print(f"\n{'='*60}")
        print(f"{log_type.upper()} REGRESSION ANALYSIS - {self.ticker}")
        print(f"{'='*60}")
        print(f"Log Slope: {model.coef_[0]:.6f}")
        print(f"Log Intercept: {model.intercept_:.4f}")
        print(f"R² Score (Log Scale): {r2:.4f}")
        print(f"RMSE (Log Scale): {rmse:.4f}")
        print(f"Daily % Change: {pct_change_per_day:.4f}%")
        print(f"Annual % Change: {pct_change_per_day * 252:.2f}%")
        
        return results
    
    def multiple_regression(self, target_col='Adj Close', plot=True):
        """
        Perform multiple regression using multiple features.
        
        Features:
        - Time trend (Days)
        - Lagged price (Price_Lag1)
        - Log price (Log_Price)
        
        Parameters:
        -----------
        target_col : str
            Column to analyze
        plot : bool
            Whether to plot results
        
        Returns:
        --------
        dict : Multiple regression statistics
        """
        df = self.prepare_regression_data(target_col)
        
        # Select features
        feature_cols = ['Days', 'Price_Lag1', 'Log_Price']
        X = df[feature_cols].dropna().values
        y = df[target_col].iloc[len(df) - len(X):].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = LinearRegression()
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        adjusted_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - len(feature_cols) - 1)
        
        results = {
            'model': model,
            'scaler': scaler,
            'feature_names': feature_cols,
            'coefficients': dict(zip(feature_cols, model.coef_)),
            'intercept': model.intercept_,
            'r2_score': r2,
            'adjusted_r2': adjusted_r2,
            'rmse': rmse,
            'y_pred': y_pred,
            'y': y,
            'X': X_scaled
        }
        
        self.results['multiple_regression'] = results
        
        if plot:
            self._plot_multiple_regression(results, target_col)
        
        print(f"\n{'='*60}")
        print(f"MULTIPLE REGRESSION ANALYSIS - {self.ticker}")
        print(f"{'='*60}")
        print(f"Features: {', '.join(feature_cols)}")
        print(f"\nCoefficients (Standardized):")
        for feat, coef in results['coefficients'].items():
            print(f"  {feat}: {coef:.6f}")
        print(f"\nIntercept: {model.intercept_:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"Adjusted R²: {adjusted_r2:.4f}")
        print(f"RMSE: ${rmse:.2f}")
        
        return results
    
    def statistical_summary(self):
        """
        Generate statistical summary of regression analyses.
        
        Returns:
        --------
        pd.DataFrame : Summary statistics
        """
        summary_data = []
        
        for analysis_name, results in self.results.items():
            summary_data.append({
                'Analysis': analysis_name,
                'R² Score': results.get('r2_score', 'N/A'),
                'RMSE': results.get('rmse', 'N/A'),
                'Observations': len(results.get('y', []))
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        print(f"\n{'='*60}")
        print("REGRESSION ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(summary_df.to_string(index=False))
        
        return summary_df
    
    # Plotting methods
    def _plot_linear_regression(self, results, col_name):
        """Plot linear regression results."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Regression plot
        axes[0, 0].scatter(results['X'], results['y'], alpha=0.5, label='Actual')
        axes[0, 0].plot(results['X'], results['y_pred'], 'r-', linewidth=2, label='Fitted Line')
        axes[0, 0].set_xlabel('Days')
        axes[0, 0].set_ylabel(f'{col_name} ($)')
        axes[0, 0].set_title(f'{self.ticker} - Linear Regression')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = results['y'] - results['y_pred']
        axes[0, 1].scatter(results['X'], residuals, alpha=0.5, color='green')
        axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Days')
        axes[0, 1].set_ylabel('Residuals ($)')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normality Check)')
        
        # Distribution of residuals
        axes[1, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Residuals ($)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Residuals')
        axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        
        plt.tight_layout()
        plt.savefig('results/linear_regression_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: results/linear_regression_analysis.png")
        plt.close()
    
    def _plot_log_regression(self, results, col_name):
        """Plot log regression results."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Log scale regression
        axes[0, 0].scatter(results['X'], results['y_log'], alpha=0.5, label='Actual (Log Scale)')
        axes[0, 0].plot(results['X'], results['y_log_pred'], 'r-', linewidth=2, label='Fitted Line')
        axes[0, 0].set_xlabel('Days')
        axes[0, 0].set_ylabel(f'{results["log_type"]}({col_name})')
        axes[0, 0].set_title(f'{self.ticker} - {results["log_type"]} Regression')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Original scale with log prediction
        axes[0, 1].scatter(results['X'], results['y'], alpha=0.5, label='Actual Price')
        axes[0, 1].plot(results['X'], results['y_original_scale'], 'r-', linewidth=2, label='Fitted Curve (Log)')
        axes[0, 1].set_xlabel('Days')
        axes[0, 1].set_ylabel(f'{col_name} ($)')
        axes[0, 1].set_title(f'{self.ticker} - Price in Original Scale')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residuals in log scale
        residuals_log = results['y_log'] - results['y_log_pred']
        axes[1, 0].scatter(results['X'], residuals_log, alpha=0.5, color='green')
        axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Days')
        axes[1, 0].set_ylabel(f'Residuals ({results["log_type"]})')
        axes[1, 0].set_title('Residual Plot (Log Scale)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Distribution of residuals
        axes[1, 1].hist(residuals_log, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel(f'Residuals ({results["log_type"]})')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Residuals (Log Scale)')
        axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        
        plt.tight_layout()
        plt.savefig('results/log_regression_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: results/log_regression_analysis.png")
        plt.close()
    
    def _plot_multiple_regression(self, results, col_name):
        """Plot multiple regression results."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Actual vs predicted
        axes[0, 0].scatter(results['y'], results['y_pred'], alpha=0.6)
        axes[0, 0].plot([results['y'].min(), results['y'].max()], 
                       [results['y'].min(), results['y'].max()], 'r--', lw=2)
        axes[0, 0].set_xlabel(f'Actual {col_name} ($)')
        axes[0, 0].set_ylabel(f'Predicted {col_name} ($)')
        axes[0, 0].set_title('Actual vs Predicted Values')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals
        residuals = results['y'] - results['y_pred']
        axes[0, 1].scatter(results['y_pred'], residuals, alpha=0.6, color='green')
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Feature importance (standardized coefficients)
        coef_names = list(results['coefficients'].keys())
        coef_values = list(results['coefficients'].values())
        colors = ['green' if x > 0 else 'red' for x in coef_values]
        axes[1, 0].barh(coef_names, coef_values, color=colors, alpha=0.7)
        axes[1, 0].set_xlabel('Coefficient Value')
        axes[1, 0].set_title('Feature Importance (Standardized Coefficients)')
        axes[1, 0].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # Distribution of residuals
        axes[1, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Residuals ($)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Residuals')
        axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        
        plt.tight_layout()
        plt.savefig('results/multiple_regression_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: results/multiple_regression_analysis.png")
        plt.close()


if __name__ == "__main__":
    # Load sample data
    try:
        data = pd.read_csv('data/sp500_stocks.csv', index_col=0, parse_dates=True)
        
        # Get a single stock (first ticker in the data)
        if 'Ticker' in data.columns:
            ticker = data['Ticker'].iloc[0]
            stock_data = data[data['Ticker'] == ticker][['Adj Close', 'Volume']]
        else:
            # If no ticker column, take first available columns
            stock_data = data.iloc[:, :2]
            ticker = "STOCK"
        
        print(f"Analyzing {ticker}...")
        
        # Initialize analyzer
        analyzer = RegressionAnalyzer(stock_data, ticker=ticker)
        
        # Run all analyses
        analyzer.linear_regression()
        analyzer.log_regression(base=10)
        analyzer.log_regression(base=np.e)
        analyzer.multiple_regression()
        
        # Summary statistics
        analyzer.statistical_summary()
        
    except FileNotFoundError:
        print("Data file not found. Run download_data.py first.")
        print("\nExample:")
        print("  python download_data.py")
