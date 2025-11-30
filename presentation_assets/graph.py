import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def generate_layer5():
    print("Generating Layer 5 Chart...")
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='B')
    price = 100 * np.exp(np.random.normal(0.0005, 0.02, 200).cumsum())
    df = pd.DataFrame({'Date': dates, 'Close': price})
    df.set_index('Date', inplace=True)

    # Layer 5 Logic: Forecasting
    window = 60
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Expected_Return_60d'] = df['Log_Return'].rolling(window=window).mean() * 252  # Annualized

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Stock Price ($)', color=color)
    ax1.plot(df.index, df['Close'], color=color, label='Price')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:orange'
    ax2.set_ylabel('Forecasted Annualized Return (Rolling 60d)', color=color)
    ax2.plot(df.index, df['Expected_Return_60d'], color=color, linestyle='--', label='Forecasted Return')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Layer 5: Forecasting Layer - Price vs Expected Return')
    fig.tight_layout()
    
    output_path = 'layer_5_forecast_example.png'
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")

def generate_layer6():
    print("Generating Layer 6 Chart...")
    np.random.seed(42)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
    n_assets = len(tickers)
    
    # Generate random returns with some structure
    returns = np.random.normal(0, 0.01, (100, n_assets))
    df_returns = pd.DataFrame(returns, columns=tickers)
    
    # Add some correlation
    df_returns['MSFT'] += df_returns['AAPL'] * 0.5
    df_returns['GOOGL'] += df_returns['AMZN'] * 0.6
    
    corr_matrix = df_returns.corr()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation Coefficient')
    
    plt.xticks(range(len(tickers)), tickers, rotation=45)
    plt.yticks(range(len(tickers)), tickers)
    
    # Loop over data dimensions and create text annotations.
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            text = plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                           ha="center", va="center", color="black")
                           
    plt.title('Layer 6: Portfolio Construction - Asset Correlation Matrix')
    plt.tight_layout()
    
    output_path = 'layer_6_correlation_matrix.png'
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")

def generate_layer7():
    print("Generating Layer 7 Chart...")
    np.random.seed(42)
    n_assets = 5
    mean_returns = np.array([0.12, 0.15, 0.10, 0.08, 0.20])
    # Create a positive semi-definite covariance matrix
    A = np.random.rand(n_assets, n_assets)
    cov_matrix = np.dot(A, A.T) * 0.05
    
    n_portfolios = 2000
    results = np.zeros((3, n_portfolios))
    
    for i in range(n_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)
        
        p_return = np.dot(weights, mean_returns)
        p_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        results[0,i] = p_std
        results[1,i] = p_return
        results[2,i] = results[1,i] / results[0,i]  # Sharpe Ratio

    plt.figure(figsize=(10, 6))
    plt.scatter(results[0,:], results[1,:], c=results[2,:], cmap='viridis', marker='o', s=10, alpha=0.5)
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility (Risk)')
    plt.ylabel('Expected Return')
    plt.title('Layer 7: Optimization - Efficient Frontier')

    # Highlight "Optimal" portfolio (max Sharpe)
    max_sharpe_idx = np.argmax(results[2,:])
    plt.scatter(results[0,max_sharpe_idx], results[1,max_sharpe_idx], c='red', marker='*', s=200, label='Optimal Portfolio')
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_path = 'layer_7_efficient_frontier.png'
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")

def generate_layer1():
    print("Generating Layer 1 Chart...")
    # Layer 1: User Input - Visualizing the inputs
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    
    # Draw a mock UI
    rect = plt.Rectangle((0.1, 0.1), 0.8, 0.8, fill=True, color='#f0f2f6', alpha=0.5, transform=ax.transAxes)
    ax.add_patch(rect)
    
    # Add text for inputs
    plt.text(0.5, 0.85, "User Input Layer", ha='center', va='center', fontsize=16, weight='bold')
    
    inputs = [
        ("Strategy", "Momentum"),
        ("Capital", "$100,000"),
        ("Risk Level", "Medium"),
        ("Sectors", "Technology, Finance"),
        ("Rebalance", "Monthly")
    ]
    
    for i, (label, value) in enumerate(inputs):
        y_pos = 0.7 - (i * 0.12)
        plt.text(0.2, y_pos, f"{label}:", fontsize=12, weight='bold')
        
        # Draw input box
        box = plt.Rectangle((0.45, y_pos - 0.04), 0.4, 0.08, fill=True, color='white', ec='gray')
        ax.add_patch(box)
        plt.text(0.47, y_pos, value, fontsize=12, va='center')

    output_path = 'layer_1_user_input.png'
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")

def generate_layer2():
    print("Generating Layer 2 Chart...")
    # Layer 2: Data Layer - Raw OHLCV
    np.random.seed(100)
    dates = pd.date_range(start='2024-01-01', periods=50, freq='B')
    price = 150 + np.cumsum(np.random.randn(50))
    volume = np.random.randint(1000, 5000, 50)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.plot(dates, price, label='Close Price', color='black')
    ax1.set_ylabel('Price ($)')
    ax1.set_title('Layer 2: Data Layer - Raw OHLCV Data')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.bar(dates, volume, color='gray', alpha=0.5, label='Volume')
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    
    output_path = 'layer_2_data_layer.png'
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")

def generate_layer3():
    print("Generating Layer 3 Chart...")
    # Layer 3: Hard Filter - Funnel
    stages = ['Total Universe', 'Sector Filter', 'Market Cap Filter']
    values = [5000, 1200, 800]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(stages))
    ax.barh(y_pos, values, align='center', color=['#e0e0e0', '#b0b0b0', '#808080'])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(stages)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Number of Stocks')
    ax.set_title('Layer 3: Hard Filter Layer - Universe Reduction')
    
    for i, v in enumerate(values):
        ax.text(v + 50, i, str(v), va='center', weight='bold')
        
    output_path = 'layer_3_hard_filter.png'
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")

def generate_layer4():
    print("Generating Layer 4 Chart...")
    # Layer 4: Strategy Layer - Ranking
    tickers = ['NVDA', 'AMD', 'INTC', 'QCOM', 'TXN']
    scores = [95, 88, 62, 78, 70]
    
    df = pd.DataFrame({'Ticker': tickers, 'Score': scores}).sort_values('Score', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.barh(df['Ticker'], df['Score'], color='tab:blue')
    ax.set_xlabel('Strategy Score (Momentum)')
    ax.set_title('Layer 4: Strategy Layer - Stock Ranking')
    
    # Highlight top picks
    bars[-1].set_color('tab:green')
    bars[-2].set_color('tab:green')
    
    for i, v in enumerate(df['Score']):
        ax.text(v - 5, i, str(v), va='center', color='white', weight='bold')
        
    output_path = 'layer_4_strategy_ranking.png'
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")

def generate_layer8():
    print("Generating Layer 8 Chart...")
    # Layer 8: Risk Management - VaR Histogram
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.01, 1000)
    
    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean()
    
    plt.figure(figsize=(10, 6))
    plt.hist(returns, bins=50, color='skyblue', edgecolor='black', alpha=0.7, density=True)
    
    plt.axvline(var_95, color='red', linestyle='--', linewidth=2, label=f'VaR (95%): {var_95:.2%}')
    plt.axvline(cvar_95, color='darkred', linestyle='-', linewidth=2, label=f'CVaR (95%): {cvar_95:.2%}')
    
    plt.title('Layer 8: Risk Management - Value at Risk (VaR) & CVaR')
    plt.xlabel('Daily Portfolio Returns')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = 'layer_8_risk_management.png'
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")

def generate_layer9():
    print("Generating Layer 9 Chart...")
    # Layer 9: Stress Testing - Scenario Analysis
    scenarios = ['Base Case', 'Market Crash (-20%)', 'Tech Bubble Burst', 'Interest Rate Hike', 'Pandemic Shock']
    impact = [0.05, -0.18, -0.12, -0.08, -0.15]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['green' if x >= 0 else 'red' for x in impact]
    bars = ax.bar(scenarios, impact, color=colors, alpha=0.7)
    
    ax.axhline(0, color='black', linewidth=1)
    ax.set_ylabel('Portfolio Impact (Return)')
    ax.set_title('Layer 9: Stress Testing - Portfolio Impact under Scenarios')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    
    for bar in bars:
        height = bar.get_height()
        label_y = height if height > 0 else height - 0.02
        ax.text(bar.get_x() + bar.get_width()/2., label_y,
                f'{height:.1%}',
                ha='center', va='bottom' if height > 0 else 'top', weight='bold')
                
    output_path = 'layer_9_stress_testing.png'
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")

def generate_layer10():
    print("Generating Layer 10 Chart...")
    # Layer 10: Backtesting - Equity Curve
    np.random.seed(101)
    dates = pd.date_range(start='2023-01-01', periods=252, freq='B')
    
    # Simulate Portfolio vs Benchmark
    port_returns = np.random.normal(0.0008, 0.012, 252)
    bench_returns = np.random.normal(0.0005, 0.010, 252)
    
    port_equity = 100 * (1 + port_returns).cumprod()
    bench_equity = 100 * (1 + bench_returns).cumprod()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(dates, port_equity, label='Strategy Portfolio', color='blue', linewidth=2)
    ax.plot(dates, bench_equity, label='Benchmark (S&P 500)', color='gray', linestyle='--', alpha=0.7)
    
    ax.set_ylabel('Portfolio Value (Indexed to 100)')
    ax.set_title('Layer 10: Backtesting - Historical Performance Simulation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Fill area under curve for visual appeal
    ax.fill_between(dates, port_equity, 100, where=(port_equity >= 100), color='blue', alpha=0.1)
    ax.fill_between(dates, port_equity, 100, where=(port_equity < 100), color='red', alpha=0.1)
    
    output_path = 'layer_10_backtesting.png'
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")

def generate_layer11():
    print("Generating Layer 11 Chart...")
    # Layer 11: Signal Generation - Trade Instructions
    data = {
        'Ticker': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
        'Signal': ['REBALANCE', 'BUY', 'SELL', 'HOLD', 'REBALANCE'],
        'Old Weight': ['12.5%', '0.0%', '8.2%', '5.0%', '10.0%'],
        'New Weight': ['14.0%', '5.0%', '0.0%', '5.0%', '8.5%'],
        'Action': ['Buy +1.5%', 'Buy +5.0%', 'Sell All', 'No Change', 'Sell -1.5%']
    }
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    ax.axis('tight')
    
    # Create table
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Color code signals
    for i in range(len(df)):
        signal = df.iloc[i]['Signal']
        color = 'white'
        if signal == 'BUY': color = '#d4edda'  # Light green
        elif signal == 'SELL': color = '#f8d7da'  # Light red
        elif signal == 'REBALANCE': color = '#fff3cd'  # Light yellow
        
        for j in range(len(df.columns)):
            table[(i+1, j)].set_facecolor(color)
            
    # Header style
    for j in range(len(df.columns)):
        table[(0, j)].set_facecolor('#e9ecef')
        table[(0, j)].set_text_props(weight='bold')
        
    plt.title('Layer 11: Signal Generation - Trade Instructions', pad=20, fontsize=14, weight='bold')
    
    output_path = 'layer_11_signal_generation.png'
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")

def generate_layer12():
    print("Generating Layer 12 Chart...")
    # Layer 12: Execution - Transaction Log
    data = {
        'Date': ['2024-01-02', '2024-01-02', '2024-01-02', '2024-02-01', '2024-02-01'],
        'Ticker': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN'],
        'Side': ['BUY', 'BUY', 'SELL', 'SELL', 'BUY'],
        'Shares': ['142', '55', '20', '10', '45'],
        'Price': ['$150.00', '$300.00', '$2800.00', '$900.00', '$3300.00'],
        'Comm.': ['-$1.00', '-$1.00', '-$1.00', '-$1.00', '-$1.00']
    }
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    ax.axis('tight')
    
    # Create table
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Header style
    for j in range(len(df.columns)):
        table[(0, j)].set_facecolor('#e9ecef')
        table[(0, j)].set_text_props(weight='bold')
        
    # Color code Side
    for i in range(len(df)):
        side = df.iloc[i]['Side']
        color = '#d4edda' if side == 'BUY' else '#f8d7da'
        table[(i+1, 2)].set_facecolor(color)

    plt.title('Layer 12: Execution - Transaction Log (Fixed Commission)', pad=20, fontsize=14, weight='bold')
    
    output_path = 'layer_12_execution.png'
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")

def generate_layer13():
    print("Generating Layer 13 Chart...")
    # Layer 13: Performance Layer - Drawdown Analysis
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=252, freq='B')
    returns = np.random.normal(0.0005, 0.012, 252)
    cumulative = (1 + returns).cumprod()
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Top: Cumulative Return
    ax1.plot(dates, cumulative * 100, label='Portfolio Value', color='blue')
    ax1.set_ylabel('Value ($)')
    ax1.set_title('Layer 13: Performance Layer - Growth & Drawdown Analysis')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Bottom: Drawdown
    ax2.fill_between(dates, drawdown, 0, color='red', alpha=0.3, label='Drawdown')
    ax2.plot(dates, drawdown, color='red', linewidth=1)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    
    # Add metrics box
    metrics_text = (
        f"Total Return: {(cumulative[-1]-1):.1%}\n"
        f"Max Drawdown: {drawdown.min():.1%}\n"
        f"Sharpe Ratio: 1.85\n"
        f"Sortino Ratio: 2.40"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax1.text(0.02, 0.95, metrics_text, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    output_path = 'layer_13_performance.png'
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")

def generate_layer14():
    print("Generating Layer 14 Chart...")
    # Layer 14: Rebalancing - Drift Analysis
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    target_weights = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
    
    # Simulate drift
    current_weights = np.array([0.24, 0.21, 0.18, 0.15, 0.22])
    
    drift = current_weights - target_weights
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(tickers))
    
    # Plot Targets (Gray outline)
    ax.barh(y_pos, target_weights * 100, color='none', edgecolor='gray', linestyle='--', label='Target Weight')
    
    # Plot Current (Colored by drift)
    colors = ['red' if abs(d) > 0.03 else 'green' for d in drift]
    bars = ax.barh(y_pos, current_weights * 100, color=colors, alpha=0.7, label='Current Weight')
    
    # Mark threshold
    ax.axvline(23, color='orange', linestyle=':', alpha=0.5)
    ax.axvline(17, color='orange', linestyle=':', alpha=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tickers)
    ax.invert_yaxis()
    ax.set_xlabel('Allocation (%)')
    ax.set_title('Layer 14: Rebalancing - Portfolio Drift Monitor')
    ax.legend()
    ax.grid(True, axis='x', alpha=0.3)
    
    # Annotate drift
    for i, (curr, dr) in enumerate(zip(current_weights, drift)):
        label = f"{curr:.1%} (Drift: {dr:+.1%})"
        ax.text(curr * 100 + 0.5, i, label, va='center', weight='bold')
        
        if abs(dr) > 0.03:
            ax.text(25, i, "REBALANCE NEEDED", color='red', weight='bold', va='center')

    output_path = 'layer_14_rebalancing.png'
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")

if __name__ == "__main__":
    generate_layer1()
    generate_layer2()
    generate_layer3()
    generate_layer4()
    generate_layer5()
    generate_layer6()
    generate_layer7()
    generate_layer8()
    generate_layer9()
    generate_layer10()
    generate_layer11()
    generate_layer12()
    generate_layer13()
    generate_layer14()
