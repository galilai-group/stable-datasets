"""Generate teaser figures for the SP500 and Forex datasets."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from stable_datasets.timeseries import SP500, Forex


OUTPUT_DIR = Path("docs/source/datasets/teasers")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_sp500():
    """Plot historical closing prices for selected stocks."""
    dataset = SP500(split="train")
    frame = pd.DataFrame(dataset)

    fig, ax = plt.subplots(figsize=(9, 4))

    for ticker in ("AAPL", "MSFT", "AMZN"):
        stock = frame.loc[frame["ticker"] == ticker].copy()
        stock["date"] = pd.to_datetime(stock["date"])
        stock = stock.sort_values("date")
        ax.plot(stock["date"], stock["close"], label=ticker)

    ax.set_title("S&P 500: Historical Stock Closing Prices")
    ax.set_xlabel("Date")
    ax.set_ylabel("Closing price (USD)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sp500_teaser.png", dpi=150)
    plt.close(fig)


def plot_forex():
    """Plot historical EUR/USD closing exchange rates."""
    dataset = Forex(split="train")
    frame = pd.DataFrame(dataset)

    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame = frame.sort_values("date")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(frame["date"], frame["close"], label="EUR/USD")

    ax.set_title("Forex: Historical EUR/USD Exchange Rate")
    ax.set_xlabel("Date")
    ax.set_ylabel("EUR/USD exchange rate")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "forex_teaser.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_sp500()
    plot_forex()
    print(f"Saved teaser figures to {OUTPUT_DIR}")
