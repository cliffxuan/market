import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from pathlib import Path

columns = ["Date", "Open", "High", "Low", "Close", "Volume"]

PWD = Path(__file__).absolute().parent

stocks = list(pd.read_csv(PWD / "data" / "spx-500.csv")["Symbol"])


def get_stock_data(ticker: str) -> tuple[pd.DataFrame | None, go.Figure]:
    if not ticker:
        return None
    df = yf.Ticker(ticker).history().reset_index()
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
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
        title=f"{ticker} Stock Price",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        template="plotly_dark",
        height=400,
    )

    return df[columns], fig


initials = get_stock_data(stocks[0])
with gr.Blocks() as demo:
    gr.Markdown("## Stock Price")
    ticker = gr.Dropdown(choices=stocks, value=stocks[0], label="Ticker")
    stock_table = gr.DataFrame(
        value=initials[0],
        headers=columns,
        label="Stock Price Data",
    )
    plot = gr.Plot(value=initials[1], label="Closing Price Chart")
    ticker.change(get_stock_data, ticker, [stock_table, plot])

demo.launch(debug=True, share=True)
