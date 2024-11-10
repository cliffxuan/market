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

# Add new constants
DATA_CLOSE = "Close Price"
DATA_VOLUME = "Volume"
DATA_OPTIONS = [DATA_CLOSE, DATA_VOLUME]
DEFAULT_DATA = DATA_CLOSE


@lru_cache(maxsize=100)
def fetch_stock_data(ticker: str, period: str) -> pd.DataFrame:
    """Fetch stock data from yfinance with caching."""
    df = yf.Ticker(ticker).history(period).reset_index()
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    return df


def format_volume(value):
    """Format volume numbers to human-readable format."""
    if value >= 1_000_000_000:  # Billions
        return f"{value/1_000_000_000:.1f}B"
    elif value >= 1_000_000:  # Millions
        return f"{value/1_000_000:.1f}M"
    elif value >= 1_000:  # Thousands
        return f"{value/1_000:.1f}K"
    return f"{value:.0f}"


def create_price_plot(
    dfs: list[tuple[pd.DataFrame, str]],
    use_log_scale: bool,
    show_returns: bool,
    data_type: str = DATA_CLOSE,
) -> go.Figure:
    """Create price plot with given configuration."""
    fig = go.Figure()

    for df, ticker in dfs:
        y_values = df[data_type.split()[0]]  # "Close" from "Close Price" or "Volume"
        if show_returns and data_type == DATA_CLOSE:
            # Calculate percentage returns from first day and round to 1 decimal
            first_price = y_values.iloc[0]
            y_values = (((y_values - first_price) / first_price) * 100).round(1)

        # Format volume values if showing volume data
        hover_template = None
        if data_type == DATA_VOLUME:
            hover_template = ticker + ": %{text}<extra></extra>"
            text = [format_volume(v) for v in y_values]
        else:
            text = None

        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=y_values,
                mode="lines",
                name=ticker,
                line=dict(width=2),
                text=text,
                hovertemplate=hover_template,
            )
        )

    # Determine y-axis title
    if data_type == DATA_CLOSE:
        y_title = DISPLAY_RETURNS if show_returns else "Price (USD)"
    else:
        y_title = "Volume"

    fig.update_layout(
        title="Stock Performance",
        xaxis_title="Date",
        yaxis_title=y_title,
        yaxis_type="log" if use_log_scale and not show_returns else "linear",
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
    data_type: str = DATA_CLOSE,
) -> tuple[pd.DataFrame | None, go.Figure]:
    """Get stock data and create plot."""
    show_returns = display_mode == DISPLAY_RETURNS
    all_dfs = []
    combined_df = pd.DataFrame()
    data_column = data_type.split()[0]  # "Close" or "Volume"

    for ticker in tickers:
        df = fetch_stock_data(ticker, period)
        df[data_column] = df[data_column].round(1)
        all_dfs.append((df, ticker))

        if combined_df.empty:
            combined_df = df[["Date", data_column]].copy()
            if show_returns and data_type == DATA_CLOSE:
                first_price = df[data_column].iloc[0]
                combined_df[data_column] = (
                    (df[data_column] - first_price) / first_price * 100
                ).round(1)
            elif data_type == DATA_VOLUME:
                combined_df[data_column] = combined_df[data_column].apply(format_volume)
            combined_df = combined_df.rename(columns={data_column: ticker})
        else:
            df_to_merge = df[["Date", data_column]].copy()
            if show_returns and data_type == DATA_CLOSE:
                first_price = df[data_column].iloc[0]
                df_to_merge[data_column] = (
                    (df[data_column] - first_price) / first_price * 100
                ).round(1)
            elif data_type == DATA_VOLUME:
                df_to_merge[data_column] = df_to_merge[data_column].apply(format_volume)
            df_to_merge = df_to_merge.rename(columns={data_column: ticker})
            combined_df = pd.merge(combined_df, df_to_merge, on="Date")

    combined_df = combined_df.sort_values("Date", ascending=False)
    fig = create_price_plot(all_dfs, use_log_scale, show_returns, data_type)
    return combined_df, fig


def update_plot(
    tickers: list[str],
    period: str,
    use_log_scale: bool = True,
    display_mode: str = DEFAULT_SHOW_RETURNS,
    data_type: str = DATA_CLOSE,
) -> go.Figure:
    """Update plot only without fetching data again."""
    show_returns = display_mode == DISPLAY_RETURNS
    all_dfs = [(fetch_stock_data(ticker, period), ticker) for ticker in tickers]
    return create_price_plot(all_dfs, use_log_scale, show_returns, data_type)


def on_display_mode_change(
    display_mode: str,
    ticker: list[str],
    period: str,
    log_scale: bool,
    data_type: str,
) -> tuple[gr.components.Component, pd.DataFrame, go.Figure]:
    """Handle display mode changes and update UI components."""
    is_returns = display_mode == DISPLAY_RETURNS
    new_df, new_plot = get_stock_data(
        ticker, period, False if is_returns else log_scale, display_mode, data_type
    )
    # Hide returns option if viewing volume
    return (
        gr.update(visible=not is_returns and data_type == DATA_CLOSE),
        new_df,
        new_plot,
    )


# Add handler for data type changes
def on_data_type_change(data_type, ticker, period, log_scale, display_mode):
    is_volume = data_type == DATA_VOLUME
    new_df, new_plot = get_stock_data(
        ticker, period, log_scale, display_mode, data_type
    )
    return (
        gr.update(visible=not is_volume),  # Hide display mode for volume
        gr.update(visible=not is_volume),  # Hide log scale for volume
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
        data_type = gr.Radio(
            choices=DATA_OPTIONS,
            value=DEFAULT_DATA,
            label="Data Type",
        )
        display_mode = gr.Radio(
            choices=DISPLAY_OPTIONS,
            value=DEFAULT_SHOW_RETURNS,
            label="Display Mode",
            visible=True,  # Will be updated based on data type
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
            visible=DEFAULT_SHOW_RETURNS != DISPLAY_RETURNS,
        )
    plot = gr.Plot(value=initials[1], label="Performance Chart")

    # Update all event handlers to include data_type
    ticker.change(
        get_stock_data,
        [ticker, period, log_scale, display_mode, data_type],
        [stock_table, plot],
    )
    period.change(
        get_stock_data,
        [ticker, period, log_scale, display_mode, data_type],
        [stock_table, plot],
    )
    log_scale.change(
        update_plot, [ticker, period, log_scale, display_mode, data_type], plot
    )
    display_mode.change(
        on_display_mode_change,
        [display_mode, ticker, period, log_scale, data_type],
        [log_scale, stock_table, plot],
    )

    data_type.change(
        on_data_type_change,
        [data_type, ticker, period, log_scale, display_mode],
        [display_mode, log_scale, stock_table, plot],
    )

demo.launch(debug=True, share=True)
