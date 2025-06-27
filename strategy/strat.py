import pandas as pd
import numpy as np

def load_merged_data(csv_path: str) -> pd.DataFrame:
    """
    Loads the merged returns + PE ratio CSV into a DataFrame.

    Parameters:
        csv_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_csv(csv_path, parse_dates=["reportDate"])


def normalize_pe_ratios_rowwise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Caps PE ratios at 100, then performs row-wise standard normalization (z-score)
    on all PE ratio columns (those prefixed with 'pe_').

    Parameters:
        df (pd.DataFrame): DataFrame containing 'pe_' prefixed columns.

    Returns:
        pd.DataFrame: Copy of df with PE ratio columns capped and row-wise normalized.
    """
    pe_cols = [col for col in df.columns if col.startswith("pe_")]
    
    # Extract and cap PE values
    pe_data = df[pe_cols].clip(upper=50)

    # Row-wise normalization
    row_means = pe_data.mean(axis=1)
    row_stds = pe_data.std(axis=1).replace(0, pd.NA)  # avoid division by zero

    normalized_pe = (pe_data.sub(row_means, axis=0)).div(row_stds, axis=0)

    # Replace PE columns in the original dataframe
    df_normalized = df.copy()
    df_normalized.loc[:, pe_cols] = normalized_pe

    return df_normalized


def calculate_growth_strategy_weights(df: pd.DataFrame, max_weight: float = 0.10) -> pd.DataFrame:
    """
    Calculates strategy weights based on standardized PE ratios (z-scores).
    Applies softmax transformation to emphasize higher values, and limits idiosyncratic risk.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'pe_' prefixed standardized PE columns.
        max_weight (float): Maximum allowed weight per stock (default = 10%).

    Returns:
        pd.DataFrame: DataFrame of strategy weights (same shape as PE columns, but without 'pe_' prefix).
    """
    pe_cols = [col for col in df.columns if col.startswith("pe_")]
    tickers = [col.replace("pe_", "") for col in pe_cols]

    weights_df = pd.DataFrame(columns=tickers, index=df.index)

    for idx, row in df[pe_cols].iterrows():
        # Softmax transformation
        z = row.fillna(-np.inf)
        exp_z = np.exp(z - np.nanmax(z))  # for numerical stability
        weights = exp_z / np.nansum(exp_z)

        # Cap weight
        weights = np.minimum(weights, max_weight)
        weights = weights / weights.sum()

        # Store in DataFrame
        weights_df.loc[idx] = weights.values

    weights_df.index = df['reportDate'].values  # optional: set date index
    weights_df.reset_index(inplace=True)
    weights_df.rename(columns={"index": "reportDate"}, inplace=True)

    return weights_df

def calculate_portfolio_returns(weights_df: pd.DataFrame, returns_df: pd.DataFrame):
    """
    Calculates the strategy portfolio and equal-weighted benchmark returns,
    along with their cumulative returns (starting from 1).

    Parameters:
        weights_df (pd.DataFrame): DataFrame with 'reportDate' and stock weights (columns = tickers).
        returns_df (pd.DataFrame): DataFrame with 'reportDate' and stock returns (columns = tickers).

    Returns:
        pd.DataFrame: DataFrame with ['reportDate', 'strategy_return', 'equal_weight_return',
                                      'strategy_cumulative_return', 'equal_weight_cumulative_return'].
    """
    # Determine shared tickers
    weight_tickers = [col for col in weights_df.columns if col != "reportDate"]
    return_tickers = [col for col in returns_df.columns if col != "reportDate"]
    shared_tickers = list(set(weight_tickers).intersection(set(return_tickers)))

    # Subset to shared columns + reportDate
    weights_df = weights_df[["reportDate"] + shared_tickers].copy()
    returns_df = returns_df[["reportDate"] + shared_tickers].copy()

    # Merge returns and weights
    merged = pd.merge(weights_df, returns_df, on="reportDate", suffixes=("_w", "_r"))

    strategy_returns = []
    equal_weight_returns = []

    for _, row in merged.iterrows():
        w = {col.replace("_w", ""): row[col] for col in row.index if col.endswith("_w")}
        r = {col.replace("_r", ""): row[col] for col in row.index if col.endswith("_r")}

        shared = list(set(w.keys()).intersection(set(r.keys())))
        
        weights = pd.Series({k: w[k] for k in shared})
        returns = pd.Series({k: r[k] for k in shared})

        strat_ret = (weights * returns).sum()

        valid_returns = pd.Series([r[k] for k in shared if pd.notna(r[k])])
        if not valid_returns.empty:
            equal_weights = np.ones(len(valid_returns)) / len(valid_returns)
            eq_ret = (equal_weights * valid_returns.values).sum()
        else:
            eq_ret = np.nan

        strategy_returns.append(strat_ret)
        equal_weight_returns.append(eq_ret)

    result_df = pd.DataFrame({
        "reportDate": merged["reportDate"],
        "strategy_return": strategy_returns,
        "equal_weight_return": equal_weight_returns
    })

    # Cumulative returns starting at 1
    result_df["strategy_cumulative_return"] = (1 + result_df["strategy_return"].fillna(0)).cumprod()
    result_df["equal_weight_cumulative_return"] = (1 + result_df["equal_weight_return"].fillna(0)).cumprod()

    return result_df




csv_path = r"C:\Coding\equity_quant_strat\data_layer\interim_data\merged_returns_pe.csv"

df_merged = load_merged_data(csv_path)
df_normalized = normalize_pe_ratios_rowwise(df_merged)

weights_df = calculate_growth_strategy_weights(df_normalized, max_weight=0.2)

returns_cols = [col for col in df_merged.columns if not col.startswith("pe_") and col != "reportDate"]
returns_df = df_merged[["reportDate"] + returns_cols].copy()

# Calculate strategy and equal-weighted portfolio returns
portfolio_returns_df = calculate_portfolio_returns(weights_df, returns_df)

# Show output
print(portfolio_returns_df.head())

# Optional: Save to CSV
output_path = r"C:\Coding\equity_quant_strat\data_layer\interim_data\portfolio_vs_equal_weight.csv"
portfolio_returns_df.to_csv(output_path, index=False)
print(f"✅ Saved: {output_path}")