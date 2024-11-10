from pathlib import Path

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


def get_stock_data(ticker: str) -> tuple[pd.DataFrame | None, go.Figure]:
    df = yf.Ticker(ticker).history("10y").reset_index()
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
        title=f"{ticker} Close Price",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        template="plotly_dark",
        height=400,
    )

    return df[columns], fig  # type: ignore


initials = get_stock_data(stocks[0][1])
with gr.Blocks() as demo:
    gr.Markdown("## Stock Price")
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
    plot = gr.Plot(value=initials[1], label="Closing Price Chart")
    ticker.change(get_stock_data, ticker, [stock_table, plot])

demo.launch(debug=True, share=True)
