import pandas as pd
import os

def load_returns_and_pe(returns_path: str, pe_path: str):
    """
    Loads the returns and PE ratios CSVs into DataFrames.

    Parameters:
        returns_path (str): Path to the daily returns CSV.
        pe_path (str): Path to the PE ratio CSV (pivoted format).

    Returns:
        tuple: (returns_df, pe_df)
    """
    returns_df = pd.read_csv(returns_path, parse_dates=["reportDate"])
    pe_df = pd.read_csv(pe_path, parse_dates=["reportDate"])
    pe_df = pe_df.set_index("reportDate").sort_index()
    return returns_df, pe_df


def align_pe_to_returns_dates(returns_df: pd.DataFrame, pe_df: pd.DataFrame):
    """
    Aligns PE ratio data to match the dates in the returns DataFrame, using forward-fill.
    Drops any initial rows where all PE values are missing.

    Parameters:
        returns_df (pd.DataFrame): Returns data with 'reportDate' column (in dd/mm/yyyy format).
        pe_df (pd.DataFrame): PE ratios indexed by 'reportDate'.

    Returns:
        pd.DataFrame: PE data aligned to returns dates, forward-filled and cleaned.
    """
    # Convert and align dates
    daily_dates = pd.to_datetime(returns_df['reportDate'], dayfirst=True).drop_duplicates().sort_values()
    pe_aligned = pe_df.reindex(daily_dates, method='ffill')

    # Drop initial rows where all columns are NaN
    pe_aligned = pe_aligned.dropna(how="all")

    # Prepare final output
    pe_aligned.index.name = "reportDate"
    return pe_aligned.reset_index()

def filter_returns_by_pe_dates(returns_df: pd.DataFrame, pe_df: pd.DataFrame):
    """
    Filters the returns DataFrame to keep only dates that exist in the PE ratio DataFrame.

    Parameters:
        returns_df (pd.DataFrame): Returns data with 'reportDate' column.
        pe_df (pd.DataFrame): PE data with 'reportDate' column.

    Returns:
        pd.DataFrame: Filtered returns DataFrame.
    """
    pe_dates = pd.to_datetime(pe_df['reportDate'])
    returns_df['reportDate'] = pd.to_datetime(returns_df['reportDate'], dayfirst=True)

    filtered_returns = returns_df[returns_df['reportDate'].isin(pe_dates)].copy()
    return filtered_returns

def merge_returns_and_pe(returns_df: pd.DataFrame, pe_df: pd.DataFrame):
    """
    Merges returns and PE ratio DataFrames on 'reportDate' and renames columns accordingly.

    Parameters:
        returns_df (pd.DataFrame): Daily returns DataFrame (includes 'reportDate').
        pe_df (pd.DataFrame): Aligned PE ratio DataFrame (includes 'reportDate').

    Returns:
        pd.DataFrame: Combined DataFrame with 'reportDate', returns, and pe_ prefixed columns.
    """
    # Rename PE columns
    pe_renamed = pe_df.rename(columns={col: f"pe_{col}" for col in pe_df.columns if col != 'reportDate'})

    # Merge on 'reportDate'
    merged = pd.merge(returns_df, pe_renamed, on="reportDate", how="inner")
    return merged

def save_merged_data_to_csv(df: pd.DataFrame, output_folder: str, file_name: str = "merged_returns_pe.csv"):
    """
    Saves the merged returns + PE ratio DataFrame to a CSV file.

    Parameters:
        df (pd.DataFrame): The merged DataFrame to save.
        output_folder (str): Folder path to save the CSV.
        file_name (str): File name for the CSV (default is 'merged_returns_pe.csv').
    """
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, file_name)
    df.to_csv(output_path, index=False)
    print(f"✅ Merged data saved to: {output_path}")

if __name__ == "__main__":
    returns_path = r"C:\Coding\equity_quant_strat\data_layer\interim_data\jse_daily_returns.csv"
    pe_path = r"C:\Coding\equity_quant_strat\data_layer\interim_data\pe_ratios.csv"

    returns_df, pe_df = load_returns_and_pe(returns_path, pe_path)
    pe_df = align_pe_to_returns_dates(returns_df, pe_df)
    returns_df = filter_returns_by_pe_dates(returns_df, pe_df)
    final_df = merge_returns_and_pe(returns_df, pe_df)
    save_folder = r"C:\Coding\equity_quant_strat\data_layer\interim_data"
    save_merged_data_to_csv(final_df, save_folder)