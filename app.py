from pathlib import Path
from functools import lru_cache

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

PWD = Path(__file__).absolute().parent
stock_df = pd.read_csv(PWD / "data" / "spx-500.csv")
stocks = [
    (f"{record['Security']} ({record['Symbol']})", record["Symbol"])
    for record in stock_df.to_dict("records")
]
columns = ["Date", "Open", "High", "Low", "Close", "Volume"]


@lru_cache(maxsize=100)
def fetch_stock_data(ticker: str) -> pd.DataFrame:
    """Fetch stock data from yfinance with caching."""
    df = yf.Ticker(ticker).history("10y").reset_index()
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    return df


def create_price_plot(df: pd.DataFrame, ticker: str, use_log_scale: bool) -> go.Figure:
    """Create price plot with given configuration."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Close"],
            mode="lines",
            name="Close Price",
            line=dict(color="#2196F3", width=2),
        )
    )

    fig.update_layout(
        title=f"{ticker} Close Price",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        yaxis_type="log" if use_log_scale else "linear",
        hovermode="x unified",
        template="plotly_dark",
        height=400,
    )
    return fig


def get_stock_data(ticker: str, use_log_scale: bool = True) -> tuple[pd.DataFrame | None, go.Figure]:
    """Get stock data and create plot."""
    df = fetch_stock_data(ticker)
    fig = create_price_plot(df, ticker, use_log_scale)
    return df[columns], fig  # type: ignore


def update_plot(ticker: str, use_log_scale: bool = True) -> go.Figure:
    """Update plot only without fetching data again."""
    df = fetch_stock_data(ticker)
    return create_price_plot(df, ticker, use_log_scale)


initials = get_stock_data(stocks[0][1])
with gr.Blocks() as demo:
    gr.Markdown("## Stock Price")
    with gr.Row():
        ticker = gr.Dropdown(
            choices=stocks,
            value=stocks[0][1],
            label="Ticker",
        )
    stock_table = gr.DataFrame(
        value=initials[0],
        headers=columns,
        label="Stock Price Data",
    )
    with gr.Row():
        log_scale = gr.Checkbox(
            value=True,
            label="Use Logarithmic Scale",
        )
    plot = gr.Plot(value=initials[1], label="Closing Price Chart")
    
    # Update the change event to include the checkbox
    ticker.change(get_stock_data, [ticker, log_scale], [stock_table, plot])
    log_scale.change(update_plot, [ticker, log_scale], plot)

demo.launch(debug=True, share=True)
