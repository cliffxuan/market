from pathlib import Path
from functools import lru_cache

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

PWD = Path(__file__).absolute().parent
DEFAULT_PERIOD = "3mo"
DEFAULT_SHOW_RETURNS = True
DEFAULT_LOG_SCALE = False
stock_df = pd.read_csv(PWD / "data" / "spx-500.csv")
stocks = [
    (f"{record['Security']} ({record['Symbol']})", record["Symbol"])
    for record in stock_df.to_dict("records")
]
columns = ["Date", "Close"]


@lru_cache(maxsize=100)
def fetch_stock_data(ticker: str, period: str = "10y") -> pd.DataFrame:
    """Fetch stock data from yfinance with caching."""
    df = yf.Ticker(ticker).history(period).reset_index()
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    return df


def create_price_plot(dfs: list[tuple[pd.DataFrame, str]], use_log_scale: bool, show_returns: bool) -> go.Figure:
    """Create price plot with given configuration."""
    fig = go.Figure()
    
    for df, ticker in dfs:
        y_values = df["Close"]
        if show_returns:
            # Calculate percentage returns from first day
            first_price = y_values.iloc[0]
            y_values = ((y_values - first_price) / first_price) * 100

        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=y_values,
                mode="lines",
                name=ticker,
                line=dict(width=2),
            )
        )

    fig.update_layout(
        title="Stock Performance",
        xaxis_title="Date",
        yaxis_title="Returns (%)" if show_returns else "Price (USD)",
        yaxis_type="log" if use_log_scale and not show_returns else "linear",  # Disable log scale for returns
        hovermode="x unified",
        template="plotly_dark",
        height=400,
    )
    return fig


def get_stock_data(tickers: list[str], period: str, use_log_scale: bool = True, show_returns: bool = False) -> tuple[pd.DataFrame | None, go.Figure]:
    """Get stock data and create plot."""
    all_dfs = []
    combined_df = pd.DataFrame()
    
    for ticker in tickers:
        df = fetch_stock_data(ticker, period)
        all_dfs.append((df, ticker))
        if combined_df.empty:
            combined_df = df[columns].copy()
            combined_df = combined_df.rename(columns={'Close': ticker})
        else:
            df_to_merge = df[columns].copy()
            df_to_merge = df_to_merge.rename(columns={'Close': ticker})
            combined_df = pd.merge(combined_df, df_to_merge, on='Date')
    
    fig = create_price_plot(all_dfs, use_log_scale, show_returns)
    return combined_df, fig


def update_plot(tickers: list[str], period: str, use_log_scale: bool = True, show_returns: bool = False) -> go.Figure:
    """Update plot only without fetching data again."""
    all_dfs = [(fetch_stock_data(ticker, period), ticker) for ticker in tickers]
    return create_price_plot(all_dfs, use_log_scale, show_returns)


# Define period choices
periods = ["5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]

initials = get_stock_data(
    [stock[1] for stock in stocks[:3]], 
    DEFAULT_PERIOD, 
    use_log_scale=DEFAULT_LOG_SCALE,
    show_returns=DEFAULT_SHOW_RETURNS
)
with gr.Blocks() as demo:
    gr.Markdown("## Stock Price")
    with gr.Row():
        ticker = gr.Dropdown(
            choices=stocks,
            value=[stock[1] for stock in stocks[:3]],
            label="Ticker",
            multiselect=True,
            max_choices=5,
        )
    with gr.Row():
        period = gr.Radio(
            choices=periods,
            value=DEFAULT_PERIOD,
            label="Time Period",
        )
    stock_table = gr.DataFrame(
        value=initials[0],
        headers=columns,
        label="Stock Price Data",
    )
    with gr.Row():
        show_returns = gr.Checkbox(
            value=DEFAULT_SHOW_RETURNS,
            label="Show Returns (%)",
        )
        log_scale = gr.Checkbox(
            value=DEFAULT_LOG_SCALE,
            label="Use Logarithmic Scale",
        )
    plot = gr.Plot(value=initials[1], label="Performance Chart")
    
    # Update the change events to include period
    ticker.change(get_stock_data, [ticker, period, log_scale, show_returns], [stock_table, plot])
    period.change(get_stock_data, [ticker, period, log_scale, show_returns], [stock_table, plot])
    log_scale.change(update_plot, [ticker, period, log_scale, show_returns], plot)
    show_returns.change(update_plot, [ticker, period, log_scale, show_returns], plot)

demo.launch(debug=True, share=True)
