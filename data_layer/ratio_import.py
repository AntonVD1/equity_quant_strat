import pandas as pd
import os

import pandas as pd
import os

def extract_ratio_data_from_csvs(folder_path):
    """
    Extracts and pivots ratio data from CSVs in a folder.

    Parameters:
        folder_path (str): Folder path containing CSV files.

    Returns:
        pd.DataFrame: Pivoted DataFrame with reportDate as index, symbols as columns.
    """
    all_data = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            try:
                df = pd.read_csv(file_path)

                if {'symbol', 'reportDate', 'value'}.issubset(df.columns):
                    df = df[['symbol', 'reportDate', 'value']].copy()
                    df['reportDate'] = pd.to_datetime(df['reportDate']).dt.date
                    all_data.append(df)
                else:
                    print(f"Skipping {filename}: Missing required columns.")

            except Exception as e:
                print(f"Failed to process {filename}: {e}")

    if not all_data:
        raise ValueError("No valid CSVs found or none contain required columns.")

    combined = pd.concat(all_data, ignore_index=True)

    pivoted = combined.pivot_table(
        index='reportDate',
        columns='symbol',
        values='value',
        aggfunc='mean'
    ).sort_index()

    return pivoted


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fills missing values in a pivoted DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame with time index and symbol columns.

    Returns:
        pd.DataFrame: Forward-filled DataFrame with reportDate reset as a column.
    """
    df_filled = df.ffill()
    return df_filled.reset_index()


def cutoff_date_filter(df: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    """
    Filters a DataFrame to only include rows on or after a given cutoff date.

    Parameters:
        df (pd.DataFrame): DataFrame with a 'reportDate' column or DateTime index.
        cutoff (str): Date string in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    cutoff_date = pd.to_datetime(cutoff)

    # If 'reportDate' is a column
    if 'reportDate' in df.columns:
        df['reportDate'] = pd.to_datetime(df['reportDate'])
        return df[df['reportDate'] >= cutoff_date].copy()

    # If it's an index
    elif isinstance(df.index, pd.DatetimeIndex) or isinstance(df.index[0], pd.Timestamp):
        return df[df.index >= cutoff_date].copy()

    else:
        raise ValueError("DataFrame must have a 'reportDate' column or a datetime-like index.")


def save_pe_ratios_to_csv(df: pd.DataFrame, output_folder: str, file_name: str = "pe_ratios.csv"):
    """
    Saves the PE ratio DataFrame to CSV at the given location.

    Parameters:
        df (pd.DataFrame): Pivoted DataFrame to save.
        output_folder (str): Destination folder path.
        file_name (str): Desired output file name.
    """
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, file_name)
    df.to_csv(output_path, index=True)
    print(f"✅ PE ratios saved to: {output_path}")


if __name__ == "__main__":
    folder = r"C:\Coding\equity_quant_strat\data_files\files"
    save_folder = r"C:\Coding\equity_quant_strat\data_layer\interim_data"
    pivoted_df = extract_ratio_data_from_csvs(folder)
    final_df = fill_missing_values(pivoted_df)
    final_df = cutoff_date_filter(final_df, "2007-01-01")
    save_pe_ratios_to_csv(final_df, save_folder)
    print(final_df.columns.tolist())
    print(final_df.head())
