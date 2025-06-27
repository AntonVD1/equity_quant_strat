
import os
import yfinance as yf
import pandas as pd

def to_yahoo_ticker(symbol):
    """
    Converts from 'ang:sj' to 'ANG.JO' for yfinance.
    """
    return symbol.split(":")[0].upper() + ".JO"


def get_jse_daily_returns(symbols, start_date="2007-01-01"):
    """
    Downloads daily stock prices and computes daily returns from Yahoo Finance.

    Parameters:
        symbols (list): List of JSE symbols in 'xxx:sj' format, starting with 'reportDate'.
        start_date (str): Start date for historical data (YYYY-MM-DD).

    Returns:
        pd.DataFrame: DataFrame of daily returns with each stock as a column.
    """
    yahoo_symbols = [to_yahoo_ticker(sym) for sym in symbols if sym != 'reportDate']
    returns_df = pd.DataFrame()

    for sym, ysym in zip(symbols[1:], yahoo_symbols):
        print(f"Downloading {ysym}...")

        try:
            data = yf.download(ysym, start=start_date, interval="1d", progress=False)

            if 'Close' not in data or data.empty:
                print(f"  ❗ {ysym} has no 'Close' data or returned empty, skipping.")
                continue

            # Use Close (adjusted by default since yfinance >=0.2.0)
            data = data[['Close']].rename(columns={'Close': sym})
            data[sym] = data[sym].pct_change()

            if returns_df.empty:
                returns_df = data
            else:
                returns_df = returns_df.join(data, how='outer')

        except Exception as e:
            print(f"  ❗ Error downloading {ysym}: {e}")

    returns_df = returns_df.dropna(how='all')  # drop rows with all NaNs
    returns_df.index.name = "reportDate"
    return returns_df.reset_index()

def save_returns_to_csv(df: pd.DataFrame, output_folder: str, file_name: str = "jse_daily_returns.csv"):
    """
    Saves the given DataFrame to a CSV file in the specified folder.
    """
    import os

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, file_name)

    df.to_csv(output_path, index=False)  # ✅ correct
    print(f"✅ Saved returns CSV to: {output_path}")

symbols = ['reportDate', 'ang:sj', 'apn:sj', 'ari:sj', 'avi:sj', 'baw:sj', 'bid:sj', 'bvt:sj', 
           'cls:sj', 'cpi:sj', 'dcp:sj', 'drd:sj', 'dsy:sj', 'exx:sj', 'fsr:sj', 'gfi:sj', 
           'grt:sj', 'imp:sj', 'mrp:sj', 'mtm:sj', 'mtn:sj', 'ned:sj', 'npn:sj', 'ny1:sj', 
           'omu:sj', 'out:sj', 'pph:sj', 'rdf:sj', 'rem:sj', 'sbk:sj', 'shp:sj', 'snt:sj', 
           'sol:sj', 'ssw:sj', 'tbs:sj']

returns_df = get_jse_daily_returns(symbols, start_date="2007-01-01")

save_folder = r"C:\Coding\equity_quant_strat\data_layer\interim_data"
save_returns_to_csv(returns_df, output_folder=save_folder)