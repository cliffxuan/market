from pathlib import Path
from functools import lru_cache

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

PWD = Path(__file__).absolute().parent
PERIODS = ["5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
DISPLAY_RETURNS = "Returns (%)"
DISPLAY_OPTIONS = [DISPLAY_RETURNS, "Price (USD)"]
DEFAULT_PERIOD = PERIODS[2]
DEFAULT_SHOW_RETURNS = DISPLAY_OPTIONS[0]
DEFAULT_LOG_SCALE = False
DEFAULT_INITIAL_STOCKS = 10
MAX_STOCKS = 20
stock_df = pd.read_csv(PWD / "data" / "spx-500.csv")
stocks = [
    (f"{record['Security']} ({record['Symbol']})", record["Symbol"])
    for record in stock_df.to_dict("records")
]
columns = ["Date", "Close"]


@lru_cache(maxsize=100)
def fetch_stock_data(ticker: str, period: str) -> pd.DataFrame:
    """Fetch stock data from yfinance with caching."""
    df = yf.Ticker(ticker).history(period).reset_index()
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    return df


def create_price_plot(
    dfs: list[tuple[pd.DataFrame, str]], use_log_scale: bool, show_returns: bool
) -> go.Figure:
    """Create price plot with given configuration."""
    fig = go.Figure()

    for df, ticker in dfs:
        y_values = df["Close"]
        if show_returns:
            # Calculate percentage returns from first day and round to 1 decimal
            first_price = y_values.iloc[0]
            y_values = (((y_values - first_price) / first_price) * 100).round(1)

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
        yaxis_title=DISPLAY_RETURNS if show_returns else "Price (USD)",
        yaxis_type="log"
        if use_log_scale and not show_returns
        else "linear",  # Disable log scale for returns
        hovermode="x unified",
        template="plotly_dark",
        height=400,
    )
    return fig


def get_stock_data(
    tickers: list[str],
    period: str,
    use_log_scale: bool = True,
    display_mode: str = DEFAULT_SHOW_RETURNS,
) -> tuple[pd.DataFrame | None, go.Figure]:
    """Get stock data and create plot."""
    show_returns = display_mode == DISPLAY_RETURNS
    all_dfs = []
    combined_df = pd.DataFrame()

    for ticker in tickers:
        df = fetch_stock_data(ticker, period)
        df["Close"] = df["Close"].round(1)
        all_dfs.append((df, ticker))

        if combined_df.empty:
            combined_df = df[columns].copy()
            if show_returns:
                # Calculate returns for first ticker
                first_price = df["Close"].iloc[0]
                combined_df["Close"] = (
                    (df["Close"] - first_price) / first_price * 100
                ).round(1)
            combined_df = combined_df.rename(columns={"Close": ticker})
        else:
            df_to_merge = df[columns].copy()
            if show_returns:
                # Calculate returns for additional tickers
                first_price = df["Close"].iloc[0]
                df_to_merge["Close"] = (
                    (df["Close"] - first_price) / first_price * 100
                ).round(1)
            df_to_merge = df_to_merge.rename(columns={"Close": ticker})
            combined_df = pd.merge(combined_df, df_to_merge, on="Date")

    combined_df = combined_df.sort_values("Date", ascending=False)
    fig = create_price_plot(all_dfs, use_log_scale, show_returns)
    return combined_df, fig


def update_plot(
    tickers: list[str],
    period: str,
    use_log_scale: bool = True,
    display_mode: str = DEFAULT_SHOW_RETURNS,
) -> go.Figure:
    """Update plot only without fetching data again."""
    show_returns = display_mode == DISPLAY_RETURNS
    all_dfs = [(fetch_stock_data(ticker, period), ticker) for ticker in tickers]
    return create_price_plot(all_dfs, use_log_scale, show_returns)


def on_display_mode_change(
    display_mode: str, ticker: list[str], period: str, log_scale: bool
) -> tuple[gr.components.Component, pd.DataFrame, go.Figure]:
    """Handle display mode changes and update UI components."""
    is_returns = display_mode == DISPLAY_RETURNS
    # Get new data with log scale forced off for returns
    new_df, new_plot = get_stock_data(
        ticker, period, False if is_returns else log_scale, display_mode
    )
    # Return values in order matching the outputs
    return (
        gr.update(visible=not is_returns),  # Changed from interactive to visible
        new_df,
        new_plot,
    )


# Get initial stock tickers
initial_tickers = [stock[1] for stock in stocks[:DEFAULT_INITIAL_STOCKS]]
initials = get_stock_data(
    initial_tickers,
    DEFAULT_PERIOD,
    use_log_scale=DEFAULT_LOG_SCALE,
    display_mode=DEFAULT_SHOW_RETURNS,
)
with gr.Blocks() as demo:
    gr.Markdown("## Stock Price")
    with gr.Row():
        ticker = gr.Dropdown(
            choices=stocks,
            value=initial_tickers,
            label="Ticker",
            multiselect=True,
            max_choices=MAX_STOCKS,
        )
    with gr.Row():
        period = gr.Radio(
            choices=PERIODS,
            value=DEFAULT_PERIOD,
            label="Time Period",
        )
    with gr.Row():
        display_mode = gr.Radio(
            choices=DISPLAY_OPTIONS,
            value=DEFAULT_SHOW_RETURNS,
            label="Display Mode",
        )
    stock_table = gr.DataFrame(
        value=initials[0],
        headers=columns,
        label="Stock Data",
    )
    with gr.Row():
        log_scale = gr.Checkbox(
            value=DEFAULT_LOG_SCALE,
            label="Use Logarithmic Scale",
            visible=DEFAULT_SHOW_RETURNS
            != DISPLAY_RETURNS,  # Initially hidden if showing returns
        )
    plot = gr.Plot(value=initials[1], label="Performance Chart")

    # Update the change events
    ticker.change(
        get_stock_data, [ticker, period, log_scale, display_mode], [stock_table, plot]
    )
    period.change(
        get_stock_data, [ticker, period, log_scale, display_mode], [stock_table, plot]
    )
    log_scale.change(update_plot, [ticker, period, log_scale, display_mode], plot)
    display_mode.change(
        on_display_mode_change,
        [display_mode, ticker, period, log_scale],
        [log_scale, stock_table, plot],
    )

demo.launch(debug=True, share=True)
