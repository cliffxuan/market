import gradio as gr
import yfinance as yf
import pandas as pd

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history().reset_index()
    return hist[['Date', 'Close', 'Volume']]

def main():
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']  # Add more tickers as needed
    interface = gr.Interface(
        fn=get_stock_data,
        inputs=gr.Dropdown(choices=tickers, label="Select Stock Ticker"),
        outputs=gr.Dataframe(label="Stock Prices"),
        title="Stock Price Viewer",
        description="Select a stock ticker to view the closing prices for the last 30 days."
    )
    interface.launch()

if __name__ == "__main__":
    main()