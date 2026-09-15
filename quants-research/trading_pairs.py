"""Pairs trading backtest with hedge-ratio-consistent P&L and trade-level accounting.

Key modeling assumptions
------------------------
1. Signals are formed from end-of-day prices and positions are opened/closed at that close.
   A newly opened position therefore earns P&L beginning on the next close-to-close interval.
2. The hedge ratio is estimated with rolling OLS (slope + intercept) and is frozen at trade entry.
3. Pair returns are normalized by gross notional, so a long spread is one unit of A versus beta
   units of B, scaled to 100% gross exposure.
4. Transaction costs are charged on entry and exit as a fraction of gross notional.
5. Stop-loss is based on the net P&L of the currently open trade, not percentage changes in the
   synthetic spread.
"""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from statsmodels.tsa.stattools import coint


class PairsTrader:
    def __init__(
        self,
        symbol_A,
        symbol_B,
        start_date,
        end_date,
        window=60,
        z_threshold=2.0,
        exit_z=0.0,
        stop_loss=0.10,
        transaction_cost=0.001,
    ):
        """Initialize the pairs-trading strategy.

        Parameters
        ----------
        window : int
            Rolling window used for hedge-ratio estimation and spread normalization.
        z_threshold : float
            Absolute z-score required to enter a trade.
        exit_z : float
            Exit band around zero. A long spread exits when z >= -exit_z and a short
            spread exits when z <= +exit_z. Use 0.0 for a zero-crossing exit.
        stop_loss : float
            Maximum tolerated net loss on an individual trade (e.g. 0.05 = 5%).
        transaction_cost : float
            Cost per entry or exit as a fraction of gross notional (0.001 = 10 bps).
        """
        if window < 2:
            raise ValueError("window must be at least 2")
        if z_threshold <= 0:
            raise ValueError("z_threshold must be positive")
        if exit_z < 0 or exit_z >= z_threshold:
            raise ValueError("exit_z must satisfy 0 <= exit_z < z_threshold")
        if stop_loss <= 0 or stop_loss >= 1:
            raise ValueError("stop_loss must be between 0 and 1")
        if transaction_cost < 0:
            raise ValueError("transaction_cost cannot be negative")

        self.symbol_A = symbol_A
        self.symbol_B = symbol_B
        self.start_date = start_date
        self.end_date = end_date
        self.window = window
        self.z_threshold = z_threshold
        self.exit_z = exit_z
        self.stop_loss = stop_loss
        self.transaction_cost = transaction_cost

        self.data = None
        self.results = None
        self.cointegration_pvalue = np.nan

        plt.style.use("default")
        plt.rcParams.update(
            {
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "axes.grid": True,
                "grid.alpha": 0.3,
                "grid.linestyle": "--",
                "axes.labelsize": 10,
                "axes.titlesize": 12,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "lines.linewidth": 1.5,
                "font.family": "sans-serif",
            }
        )
        sns.set_theme(style="whitegrid", palette="husl")
        sns.set_context("notebook", font_scale=1.1)

    @staticmethod
    def _extract_price_series(raw, symbol):
        """Extract adjusted close (preferred) or close from yfinance output."""
        if raw.empty:
            raise ValueError(f"No data received for {symbol}")

        preferred_fields = ("Adj Close", "Close")

        if isinstance(raw.columns, pd.MultiIndex):
            level0 = raw.columns.get_level_values(0)
            for field in preferred_fields:
                if field in level0:
                    values = raw[field]
                    if isinstance(values, pd.DataFrame):
                        if symbol in values.columns:
                            series = values[symbol]
                        else:
                            series = values.iloc[:, 0]
                    else:
                        series = values
                    return pd.to_numeric(series, errors="coerce").rename(symbol)
        else:
            for field in preferred_fields:
                if field in raw.columns:
                    return pd.to_numeric(raw[field], errors="coerce").rename(symbol)

        raise ValueError(
            f"Could not find an adjusted-close or close column for {symbol}. "
            f"Available columns: {raw.columns.tolist()}"
        )

    def fetch_data(self):
        """Download, align, validate, and store historical prices."""
        try:
            print(f"\nFetching data for {self.symbol_A} and {self.symbol_B}...")

            raw_A = yf.download(
                self.symbol_A,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=False,
                progress=False,
            )
            raw_B = yf.download(
                self.symbol_B,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=False,
                progress=False,
            )

            price_A = self._extract_price_series(raw_A, self.symbol_A)
            price_B = self._extract_price_series(raw_B, self.symbol_B)

            self.data = pd.concat(
                [
                    price_A.rename(f"{self.symbol_A}_price"),
                    price_B.rename(f"{self.symbol_B}_price"),
                ],
                axis=1,
                join="inner",
            ).dropna()

            # Two rolling windows are effectively needed before a fully normalized
            # residual z-score is available, so require at least 2 * window points.
            min_required = 2 * self.window
            if len(self.data) < min_required:
                raise ValueError(
                    f"Insufficient aligned data points ({len(self.data)}); "
                    f"need at least {min_required} for window={self.window}."
                )

            print("\nData Summary:")
            print(f"Start Date: {self.data.index[0]}")
            print(f"End Date: {self.data.index[-1]}")
            print(f"Trading Days: {len(self.data)}")
            print("\nPrice Statistics:")
            print(self.data.describe())

            return self.data

        except Exception as exc:
            print("\nDetailed error information:")
            print(f"Error type: {type(exc).__name__}")
            print(f"Error message: {exc}")
            raise RuntimeError(f"Error fetching data: {exc}") from exc

    def calculate_spread(self):
        """Estimate rolling OLS residual spread and its rolling z-score."""
        if self.data is None:
            raise ValueError("No data loaded. Run fetch_data() first.")

        price_A = self.data[f"{self.symbol_A}_price"].astype(float)
        price_B = self.data[f"{self.symbol_B}_price"].astype(float)

        # Full-sample cointegration is reported only as a diagnostic; it is not used
        # to switch trades on/off inside this backtest.
        try:
            coint_stat, p_value, critical_values = coint(price_A, price_B)
            self.cointegration_pvalue = float(p_value)
            print("\nCointegration Analysis:")
            print(f"Test Statistic: {coint_stat:.4f}")
            print(f"P-value: {p_value:.4f}")
            print("Critical Values:")
            print(f"1%: {critical_values[0]:.4f}")
            print(f"5%: {critical_values[1]:.4f}")
            print(f"10%: {critical_values[2]:.4f}")
            if p_value > 0.05:
                print("\nWarning: pair may not be cointegrated (p-value > 0.05).")
            else:
                print("\nPair shows evidence of cointegration.")
        except Exception as exc:
            print(f"\nWarning: could not perform cointegration test: {exc}")
            self.cointegration_pvalue = np.nan

        rolling_mean_A = price_A.rolling(self.window, min_periods=self.window).mean()
        rolling_mean_B = price_B.rolling(self.window, min_periods=self.window).mean()
        rolling_var_B = price_B.rolling(self.window, min_periods=self.window).var()
        rolling_cov = price_A.rolling(self.window, min_periods=self.window).cov(price_B)

        hedge_ratio = rolling_cov / rolling_var_B.replace(0, np.nan)
        intercept = rolling_mean_A - hedge_ratio * rolling_mean_B

        # Rolling OLS residual: A = alpha + beta * B + residual.
        spread = price_A - intercept - hedge_ratio * price_B

        spread_mean = spread.rolling(self.window, min_periods=self.window).mean()
        spread_std = spread.rolling(self.window, min_periods=self.window).std()
        zscore = (spread - spread_mean) / spread_std.replace(0, np.nan)

        return pd.DataFrame(
            {
                "spread": spread,
                "zscore": zscore,
                "hedge_ratio": hedge_ratio,
                "intercept": intercept,
            },
            index=self.data.index,
        )

    def generate_signals(self):
        """Create rolling signal inputs without using future observations."""
        signals = self.calculate_spread().copy()

        signals["entry_signal"] = 0
        signals.loc[signals["zscore"] <= -self.z_threshold, "entry_signal"] = 1
        signals.loc[signals["zscore"] >= self.z_threshold, "entry_signal"] = -1

        # Actual positions are filled by calculate_returns(), because stop-loss exits
        # depend on realized trade P&L rather than only on the z-score.
        return signals

    def calculate_returns(self, signals, initial_capital=100000):
        """Run a sequential, no-look-ahead trade simulation.

        The hedge ratio is frozen at entry. A long spread holds +1 share-equivalent of
        A and -beta share-equivalents of B; returns are divided by gross notional.
        """
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")

        price_A = self.data[f"{self.symbol_A}_price"].astype(float)
        price_B = self.data[f"{self.symbol_B}_price"].astype(float)
        signals = signals.copy()

        n = len(signals)
        daily_returns = np.zeros(n, dtype=float)
        end_of_day_position = np.zeros(n, dtype=float)
        held_beta = np.full(n, np.nan, dtype=float)
        exit_reason_series = np.full(n, "", dtype=object)

        current_position = 0  # +1 long spread, -1 short spread
        entry_beta = np.nan
        entry_index = None
        entry_date = None
        entry_z = np.nan
        trade_nav = None
        trades = []

        for i in range(1, n):
            day_factor = 1.0
            exited_today = False

            # 1) Earn today's close-to-close P&L from the position held overnight.
            if current_position != 0:
                prev_A = float(price_A.iloc[i - 1])
                curr_A = float(price_A.iloc[i])
                prev_B = float(price_B.iloc[i - 1])
                curr_B = float(price_B.iloc[i])

                gross_notional = abs(prev_A) + abs(entry_beta) * abs(prev_B)
                if gross_notional <= 0 or not np.isfinite(gross_notional):
                    raise RuntimeError("Invalid gross notional while calculating pair return")

                unit_spread_pnl = (curr_A - prev_A) - entry_beta * (curr_B - prev_B)
                holding_return = current_position * unit_spread_pnl / gross_notional

                # Guard against impossible compounding caused by bad input data.
                if holding_return <= -1:
                    raise RuntimeError(
                        f"Daily strategy return <= -100% on {signals.index[i]}: "
                        f"{holding_return:.4f}"
                    )

                day_factor *= 1.0 + holding_return
                trade_nav *= 1.0 + holding_return

            z = signals["zscore"].iloc[i]
            beta = signals["hedge_ratio"].iloc[i]
            date = signals.index[i]

            # 2) Decide whether an existing position exits at today's close.
            exit_reason = None
            if current_position != 0:
                trade_return_before_exit = trade_nav - 1.0

                if trade_return_before_exit <= -self.stop_loss:
                    exit_reason = "stop_loss"
                elif np.isfinite(z):
                    if current_position == 1 and z >= -self.exit_z:
                        exit_reason = "mean_reversion"
                    elif current_position == -1 and z <= self.exit_z:
                        exit_reason = "mean_reversion"

                if i == n - 1 and exit_reason is None:
                    exit_reason = "end_of_backtest"

            if exit_reason is not None:
                day_factor *= 1.0 - self.transaction_cost
                trade_nav *= 1.0 - self.transaction_cost

                trades.append(
                    {
                        "entry_date": entry_date,
                        "exit_date": date,
                        "direction": "long_spread" if current_position == 1 else "short_spread",
                        "entry_z": entry_z,
                        "exit_z": float(z) if np.isfinite(z) else np.nan,
                        "entry_hedge_ratio": entry_beta,
                        "holding_days": int(i - entry_index),
                        "trade_return": float(trade_nav - 1.0),
                        "exit_reason": exit_reason,
                    }
                )
                exit_reason_series[i] = exit_reason

                current_position = 0
                entry_beta = np.nan
                entry_index = None
                entry_date = None
                entry_z = np.nan
                trade_nav = None
                exited_today = True

            # 3) Open a new position only after today's close, so it cannot earn today's P&L.
            # Do not open on the final observation and do not immediately re-enter after an exit.
            if (
                current_position == 0
                and not exited_today
                and i < n - 1
                and np.isfinite(z)
                and np.isfinite(beta)
            ):
                new_position = 0
                if z <= -self.z_threshold:
                    new_position = 1
                elif z >= self.z_threshold:
                    new_position = -1

                if new_position != 0:
                    current_position = new_position
                    entry_beta = float(beta)
                    entry_index = i
                    entry_date = date
                    entry_z = float(z)
                    trade_nav = 1.0

                    # Entry cost is paid at today's close.
                    day_factor *= 1.0 - self.transaction_cost
                    trade_nav *= 1.0 - self.transaction_cost

            daily_returns[i] = day_factor - 1.0
            end_of_day_position[i] = current_position
            if current_position != 0:
                held_beta[i] = entry_beta

        strategy_returns = pd.Series(daily_returns, index=signals.index, name="strategy_return")
        portfolio_value = initial_capital * (1.0 + strategy_returns).cumprod()

        signals["position"] = end_of_day_position
        signals["position_size"] = end_of_day_position  # fixed 100% gross exposure
        signals["held_hedge_ratio"] = held_beta
        signals["strategy_return"] = strategy_returns
        signals["exit_reason"] = exit_reason_series

        trades_df = pd.DataFrame(trades)

        trading_days = 252
        mean_daily = float(strategy_returns.mean())
        std_daily = float(strategy_returns.std(ddof=1))
        annualized_mean = trading_days * mean_daily
        annualized_vol = np.sqrt(trading_days) * std_daily
        sharpe_ratio = annualized_mean / annualized_vol if annualized_vol > 0 else 0.0

        downside = np.minimum(strategy_returns.to_numpy(dtype=float), 0.0)
        downside_dev = np.sqrt(np.mean(np.square(downside))) * np.sqrt(trading_days)
        sortino_ratio = annualized_mean / downside_dev if downside_dev > 0 else 0.0

        running_max = portfolio_value.cummax()
        drawdown = portfolio_value / running_max - 1.0
        max_drawdown = float(drawdown.min())

        total_trades = int(len(trades_df))
        if total_trades:
            trade_returns = trades_df["trade_return"].astype(float)
            win_rate = float((trade_returns > 0).mean())
            avg_trade_duration = float(trades_df["holding_days"].mean())
            gross_profit = float(trade_returns[trade_returns > 0].sum())
            gross_loss = float(trade_returns[trade_returns < 0].sum())
            profit_factor = gross_profit / abs(gross_loss) if gross_loss < 0 else np.inf
        else:
            win_rate = 0.0
            avg_trade_duration = 0.0
            profit_factor = 0.0

        final_return = float(portfolio_value.iloc[-1] / initial_capital - 1.0)
        annualized_return = (
            (1.0 + final_return) ** (trading_days / max(len(strategy_returns) - 1, 1)) - 1.0
            if final_return > -1.0
            else -1.0
        )

        self.results = {
            "returns": strategy_returns,
            "portfolio_value": portfolio_value,
            "sharpe_ratio": float(sharpe_ratio),
            "sortino_ratio": float(sortino_ratio),
            "max_drawdown": max_drawdown,
            "signals": signals,
            "trades": trades_df,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "avg_trade_duration": avg_trade_duration,
            "profit_factor": float(profit_factor),
            "final_return": final_return,
            "annualized_return": float(annualized_return),
            "cointegration_pvalue": float(self.cointegration_pvalue),
        }
        return self.results

    def plot_results(self):
        """Create and save strategy diagnostics."""
        if self.results is None:
            raise ValueError("Run the strategy first using run_strategy()")

        signals = self.results["signals"]
        fig = plt.figure(figsize=(15, 25))
        gs = fig.add_gridspec(5, 1, height_ratios=[2, 1.5, 1, 1.5, 1.5])

        ax1 = fig.add_subplot(gs[0])
        price_A = self.data[f"{self.symbol_A}_price"] / self.data[f"{self.symbol_A}_price"].iloc[0]
        price_B = self.data[f"{self.symbol_B}_price"] / self.data[f"{self.symbol_B}_price"].iloc[0]
        ax1.plot(price_A, label=f"{self.symbol_A} (Normalized)")
        ax1.plot(price_B, label=f"{self.symbol_B} (Normalized)")
        ax1.set_title("Normalized Stock Prices", pad=15)
        ax1.legend(loc="upper left", frameon=True)

        ax2 = fig.add_subplot(gs[1])
        ax2.plot(signals["spread"], label="Rolling OLS Residual Spread", alpha=0.7)
        ax2.set_title("Spread and Z-Score", pad=15)
        ax2.set_ylabel("Spread")

        ax2_twin = ax2.twinx()
        ax2_twin.plot(signals["zscore"], label="Z-Score", linestyle="--")
        ax2_twin.axhline(self.z_threshold, alpha=0.4)
        ax2_twin.axhline(-self.z_threshold, alpha=0.4)
        ax2_twin.axhline(self.exit_z, alpha=0.25)
        ax2_twin.axhline(-self.exit_z, alpha=0.25)
        ax2_twin.set_ylabel("Z-Score")

        ax3 = fig.add_subplot(gs[2])
        ax3.step(signals.index, signals["position"], where="post", label="Position")
        ax3.set_title("Actual Trading Position", pad=15)
        ax3.set_ylabel("-1 / 0 / +1")

        ax4 = fig.add_subplot(gs[3])
        cumulative_returns = self.results["portfolio_value"] / self.results["portfolio_value"].iloc[0]
        ax4.plot(cumulative_returns)
        ax4.set_title("Cumulative Portfolio Value", pad=15)

        ax5 = fig.add_subplot(gs[4])
        drawdown = cumulative_returns / cumulative_returns.cummax() - 1.0
        ax5.fill_between(drawdown.index, drawdown, 0, alpha=0.3)
        ax5.set_title("Drawdown", pad=15)

        metrics_text = (
            f"Total Return: {self.results['final_return']:.2%}\n"
            f"Annualized Return: {self.results['annualized_return']:.2%}\n"
            f"Sharpe Ratio: {self.results['sharpe_ratio']:.2f}\n"
            f"Sortino Ratio: {self.results['sortino_ratio']:.2f}\n"
            f"Max Drawdown: {self.results['max_drawdown']:.2%}\n"
            f"Trade Win Rate: {self.results['win_rate']:.2%}\n"
            f"Completed Trades: {self.results['total_trades']}\n"
            f"Avg Holding Period: {self.results['avg_trade_duration']:.1f} trading days"
        )
        fig.text(0.15, 0.015, metrics_text, fontsize=10, bbox=dict(facecolor="white", alpha=0.8))

        plt.tight_layout(rect=[0, 0.05, 1, 1])

        figures_dir = "Figures"
        pair_dir = os.path.join(figures_dir, f"{self.symbol_A}_{self.symbol_B}")
        os.makedirs(pair_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy_file = os.path.join(pair_dir, f"strategy_summary_{timestamp}.png")
        fig.savefig(strategy_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved strategy summary: {strategy_file}")

        plt.figure(figsize=(8, 6))
        corr_data = pd.DataFrame(
            {
                self.symbol_A: self.data[f"{self.symbol_A}_price"],
                self.symbol_B: self.data[f"{self.symbol_B}_price"],
            }
        )
        sns.heatmap(corr_data.corr(), annot=True, vmin=-1, vmax=1)
        plt.title(f"Correlation Matrix: {self.symbol_A} vs {self.symbol_B}")
        corr_file = os.path.join(pair_dir, f"correlation_{timestamp}.png")
        plt.savefig(corr_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved correlation plot: {corr_file}")

        plt.figure(figsize=(10, 6))
        sns.histplot(self.results["returns"], kde=True, bins=50)
        plt.title("Strategy Returns Distribution")
        plt.xlabel("Daily Return")
        plt.ylabel("Frequency")
        returns_file = os.path.join(pair_dir, f"returns_distribution_{timestamp}.png")
        plt.savefig(returns_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved returns distribution: {returns_file}")

        if not self.results["trades"].empty:
            trades_file = os.path.join(pair_dir, f"trade_log_{timestamp}.csv")
            self.results["trades"].to_csv(trades_file, index=False)
            print(f"Saved trade log: {trades_file}")

        return pair_dir

    def run_strategy(self, initial_capital=100000):
        """Run data loading, signal generation, backtest, plotting, and reporting."""
        if self.data is None:
            self.fetch_data()

        signals = self.generate_signals()
        results = self.calculate_returns(signals, initial_capital)
        self.plot_results()

        print("\nStrategy Performance Report")
        print("=" * 48)
        print(f"Total Return: {results['final_return']:.2%}")
        print(f"Annualized Return: {results['annualized_return']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {results['sortino_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"Trade Win Rate: {results['win_rate']:.2%}")
        print(f"Completed Trades: {results['total_trades']}")
        print(f"Avg Holding Period: {results['avg_trade_duration']:.1f} trading days")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print("=" * 48)
        return results


if __name__ == "__main__":
    pairs = [
        # ETF / macro relationships
        ("XLE", "USO"),
        ("GDX", "GLD"),
        ("XLF", "VFH"),
        ("QQQ", "SPY"),
        # Consumer staples / retail
        ("KO", "PEP"),
        ("WMT", "TGT"),
        ("COST", "WMT"),
        ("PG", "CL"),
        # Financial services
        ("JPM", "BAC"),
        ("GS", "MS"),
        ("BLK", "BEN"),
        # Semiconductors / technology hardware
        ("INTC", "AMD"),
        ("NVDA", "AMD"),
        ("NVDA", "TSM"),
        ("TSM", "UMC"),
        # Energy
        ("CVX", "XOM"),
        ("PSX", "VLO"),
        # Payments / fintech
        ("V", "MA"),
        ("PYPL", "XYZ"),  # Block changed ticker from SQ to XYZ in 2025
        # Healthcare
        ("JNJ", "PFE"),
        ("MRK", "LLY"),
        ("UNH", "CVS"),
        # Airlines
        ("AAL", "UAL"),
        ("DAL", "LUV"),
        # Automakers
        ("TSLA", "GM"),
        ("F", "TM"),
        # Online / traditional retail
        ("AMZN", "WMT"),
        ("EBAY", "ETSY"),
        # Software / services
        ("MSFT", "GOOGL"),
        ("CRM", "ORCL"),
        # Consumer electronics
        ("AAPL", "SONY"),  # Sony changed ticker from SNE to SONY in 2021
        # Telecommunications
        ("T", "VZ"),
        ("TMUS", "VOD"),
        # Real estate
        ("SPG", "MAC"),
    ]

    start_date = "2020-01-01"
    end_date = "2026-08-31"
    results_summary = []

    for symbol_A, symbol_B in pairs:
        try:
            print(f"\nAnalyzing pair: {symbol_A} - {symbol_B}")
            trader = PairsTrader(
                symbol_A,
                symbol_B,
                start_date,
                end_date,
                window=30,
                z_threshold=1.5,
                exit_z=0.0,
                stop_loss=0.05,
                transaction_cost=0.001,
            )

            results = trader.run_strategy(initial_capital=100000)
            results_summary.append(
                {
                    "pair": f"{symbol_A}-{symbol_B}",
                    "cointegration_pvalue": results["cointegration_pvalue"],
                    "total_return": results["final_return"],
                    "annualized_return": results["annualized_return"],
                    "sharpe_ratio": results["sharpe_ratio"],
                    "sortino_ratio": results["sortino_ratio"],
                    "max_drawdown": results["max_drawdown"],
                    "win_rate": results["win_rate"],
                    "total_trades": results["total_trades"],
                    "avg_trade_duration": results["avg_trade_duration"],
                    "profit_factor": results["profit_factor"],
                }
            )

        except Exception as exc:
            print(f"Error analyzing {symbol_A}-{symbol_B}: {exc}")
            continue

    summary_df = pd.DataFrame(results_summary)
    if summary_df.empty:
        print("\nNo pairs completed successfully; no summary file was created.")
    else:
        print("\nAggregate Strategy Performance")
        print("=" * 80)
        metrics_to_analyze = [
            ("total_return", "Total Return", "pct"),
            ("annualized_return", "Annualized Return", "pct"),
            ("sharpe_ratio", "Sharpe Ratio", "num"),
            ("sortino_ratio", "Sortino Ratio", "num"),
            ("win_rate", "Trade Win Rate", "pct"),
        ]

        for column, display_name, fmt in metrics_to_analyze:
            values = summary_df[column].replace([np.inf, -np.inf], np.nan).dropna()
            if values.empty:
                continue
            mean_value = values.mean()
            std_value = values.std(ddof=0)
            if fmt == "pct":
                print(f"{display_name}: mean={mean_value:.2%}, std={std_value:.2%}")
            else:
                print(f"{display_name}: mean={mean_value:.2f}, std={std_value:.2f}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"pairs_trading_results_{timestamp}.csv"
        summary_df.to_csv(results_filename, index=False)
        print(f"\nNumeric results saved to: {results_filename}")

        display_df = summary_df.copy()
        for col in ["total_return", "annualized_return", "max_drawdown", "win_rate"]:
            display_df[col] = display_df[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
        for col in ["cointegration_pvalue", "sharpe_ratio", "sortino_ratio", "profit_factor"]:
            display_df[col] = display_df[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        display_df["avg_trade_duration"] = display_df["avg_trade_duration"].map(lambda x: f"{x:.1f}")

        print("\nDetailed Results Summary:")
        print(display_df.to_string(index=False))

        strong_performers = summary_df[
            (summary_df["annualized_return"] > 0)
            & (summary_df["sharpe_ratio"] > 0.1)
            & (summary_df["profit_factor"] > 1.5)
            & (summary_df["max_drawdown"] > -0.20)
        ].sort_values("sharpe_ratio", ascending=False)

        print("\nTop performing pairs based on risk-adjusted returns:")
        if strong_performers.empty:
            print("No pairs met the strong-performer filters.")
        else:
            print(
                strong_performers[
                    [
                        "pair",
                        "annualized_return",
                        "sharpe_ratio",
                        "win_rate",
                        "profit_factor",
                        "max_drawdown",
                    ]
                ].head()
            )
