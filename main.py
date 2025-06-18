# main.py
import os
from datetime import datetime, timedelta
import argparse
from typing import Optional
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
import requests
from config import (
    TICKERS,
    SELL_TICKERS,
    RSI_THRESHOLD,
    PRICE_DROP_THRESHOLD,
    RESULT_DIR,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    SELL_RSI_THRESHOLD,
    SELL_GROWTH_THRESHOLD,
    SELL_RECOMMENDATION_KEYS,
    SELL_SCORE_THRESHOLD,
)
import numpy as np
import re


def trimmed_mean(series: pd.Series, proportion: float = 0.1) -> float:
    """Return the trimmed mean of a pandas Series.

    Parameters
    ----------
    series : pd.Series or pd.DataFrame
        Input data which may occasionally be a single-column DataFrame.  In
        that case it is squeezed into a Series before processing.
    proportion : float, optional
        Fraction of data to trim from each end when computing the mean.
    """

    if isinstance(series, pd.DataFrame):
        series = series.squeeze()

    if series.empty:
        return float("nan")

    s = series.sort_values()
    cut = int(len(s) * proportion)
    if len(s) - 2 * cut <= 0:
        return float(s.mean())
    return float(s.iloc[cut: len(s) - cut].mean())

def escape_markdown(text):
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r'([_\*\[\]()~`>#+\-=|{}.!])', r'\\\1', text)

def get_stock_data(ticker):
    data = yf.download(ticker, period="2mo", interval="1d", auto_adjust=True)
    data.dropna(inplace=True)
    return data

def analyze_stock(ticker):
    df = get_stock_data(ticker)

    if df.empty or 'Close' not in df:
        return {'Ticker': ticker, 'Error': 'No data'}

    try:
        close_series = df["Close"]
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.squeeze()

        current_price = close_series.iloc[-1].item()
        month_open_price = close_series.iloc[0].item()
        price_change = ((current_price - month_open_price) / month_open_price) * 100

        # Normalized change using trimmed mean of first and second month
        half = len(close_series) // 2 or 1
        past_trimmed = trimmed_mean(close_series.iloc[:half])
        current_trimmed = trimmed_mean(close_series.iloc[half:])
        if np.isnan(past_trimmed) or past_trimmed == 0:
            normalized_change = np.nan
        else:
            normalized_change = ((current_trimmed - past_trimmed) / past_trimmed) * 100

        rsi_series = RSIIndicator(close_series, window=14).rsi()
        rsi = rsi_series.iloc[-1]

        if pd.isna(rsi):
            return {'Ticker': ticker, 'Error': 'RSI is NaN'}

        avg_30d = df['Close'].rolling(window=30).mean().iloc[-1]
        below_avg = current_price < avg_30d

        ticker_obj = yf.Ticker(ticker)
        try:
            info = ticker_obj.info
            recommendation = info.get('recommendationKey', 'n/a')
            target_mean_price = info.get('targetMeanPrice', None)
            long_name = info.get('longName', '')
            sector = info.get('sector', 'Unknown')
        except Exception:
            recommendation = 'n/a'
            target_mean_price = None
            long_name = ''
            sector = 'Unknown'

        return {
            'Ticker': ticker,
            'Company': long_name,
            'Sector': sector,
            'Current Price': round(current_price, 2),
            'Open Price (month)': round(month_open_price, 2),
            'Price Change %': round(price_change, 2),
            'Normalized Price Change %': round(normalized_change, 2) if not np.isnan(normalized_change) else None,
            'RSI': round(float(rsi), 2),
            'Below 30d Avg': below_avg,
            'RSI_check': float(rsi) < RSI_THRESHOLD,
            'Drop_check': price_change <= -PRICE_DROP_THRESHOLD,
            'Recommendation': recommendation,
            'Target Mean Price': round(target_mean_price, 2) if target_mean_price is not None else None,
            'Error': None
        }

    except Exception as e:
        return {'Ticker': ticker, 'Error': str(e)}

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'Markdown'
    }
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Telegram notification failed: {e}")

def format_sector_reports(df):
    df = df.copy()
    if 'Recommendation' in df.columns:
        df['Recommendation'] = df['Recommendation'].fillna('n/a').astype(str)
    df = df[df['Error'].isnull()] if 'Error' in df.columns else df
    if 'Sector' not in df.columns:
        df['Sector'] = 'Unknown'
    else:
        df['Sector'] = df['Sector'].fillna('Unknown')
    if df.empty:
        return ["No valid data to generate report."]
    sector_groups = df.groupby('Sector')
    messages = []

    legend_message = (
        "\U0001F4D8 *Пояснення до звіту по акціях*\n\n"
        "📊 *Оцінка сигналу купівлі:*\n"
        "- 🔥 – *Сильний сигнал*: RSI < 40, падіння ≥ 5%, рекомендація Buy/Strong Buy.\n"
        "- ✅ – *Помірний сигнал*: виконано 2 з 3 умов.\n"
        "- ⚠️ – *Слабкий сигнал*: лише 1 умова.\n"
        "- ❌ – *Немає сигналу*: не виконано жодної умови.\n\n"
        "📉 *Оцінка за RSI:*\n"
        "- 🧊 – RSI < 30: перепродана, можливий ріст.\n"
        "- 📉 – RSI < 40: потенціал до зростання.\n"
        "- ⚖️ – RSI 40–70: нейтральна зона.\n"
        "- 🔺 – RSI > 70: підвищений оптимізм.\n"
        "- 🚫 – RSI > 80: перегріта акція.\n\n"
        "⚠️ *Інші маркери ризику:*\n"
        "- 💧 – Зміна < 2%: стабільна динаміка.\n"
        "- ⚡ – Зміна > 30%: висока волатильність.\n"
        "📈 Стрілки вгору/вниз відображають напрям зміни ціни: 🔼🟢 – зростання, 🔽🔴 – падіння."
    )
    messages.append(legend_message)

    for sector, group in sector_groups:
        message = f"\U0001F4CB *Звіт по галузі: {escape_markdown(sector)}*"
        for _, row in group.iterrows():
            rec = escape_markdown(row['Recommendation'].capitalize() if row['Recommendation'] != 'n/a' else 'Без даних')
            rsi = float(row['RSI'])
            change = float(row['Price Change %'])
            direction = "🔼🟢" if change > 0 else "🔽🔴"
            norm_change = row.get('Normalized Price Change %')
            if pd.isna(norm_change):
                norm_str = 'n/a'
            else:
                norm_dir = "🔼🟢" if norm_change > 0 else "🔽🔴"
                norm_str = f"{norm_dir} {abs(round(norm_change, 2))}%"

            rsi_flag = rsi < 40
            drop_flag = change <= -5
            rec_flag = row['Recommendation'] in ['buy', 'strong_buy']
            score = sum([rsi_flag, drop_flag, rec_flag])

            if score == 3:
                emoji = '🔥'
            elif score == 2:
                emoji = '✅'
            elif score == 1:
                emoji = '⚠️'
            else:
                emoji = '❌'

            risk_emoji = ''
            if rsi > 80:
                risk_emoji += " 🚫 Перегріта"
            elif rsi < 30:
                risk_emoji += " 🧊 Перепродана"
            elif rsi < 40:
                risk_emoji += " 📉 Потенціал"
            elif 40 <= rsi <= 70:
                risk_emoji += " ⚖️ Нейтральна"
            elif rsi > 70:
                risk_emoji += " 🔺 Оптимізм"

            if change > 30:
                risk_emoji += " ⚡ Висока волатильність"
            elif abs(change) < 2:
                risk_emoji += " 💧 Стабільна"

            ticker = escape_markdown(row['Ticker'])
            name = escape_markdown(row['Company']) if row['Company'] else ''
            msg_line = (
                f"\n{emoji} *{ticker}* {name}: ${row['Current Price']} | "
                f"Зміна: {direction} {abs(row['Price Change %'])}% "
                f"(норм: {norm_str}) | RSI: {row['RSI']}{risk_emoji} | Реком: {rec}"
            )
            if row['Target Mean Price'] is not None:
                msg_line += f" | 🎯 ${row['Target Mean Price']}"
            message += msg_line

        messages.append(message)

    return messages

def top_recommendations(df, limit=5):
    df = df.copy()
    if 'Recommendation' in df.columns:
        df['Recommendation'] = df['Recommendation'].fillna('n/a').astype(str)

    df = df[df['Error'].isnull() & df['Target Mean Price'].notnull()]
    df['Score'] = (
        (df['RSI'] < 40).astype(int) +
        (df['Price Change %'] <= -5).astype(int) +
        (df['Recommendation'].isin(['buy', 'strong_buy'])).astype(int)
    )
    df = df[df['Score'] >= 2]
    df['Potential %'] = ((df['Target Mean Price'] - df['Current Price']) / df['Current Price']) * 100
    df = df.sort_values(by=['Score', 'Potential %'], ascending=[False, False]).head(limit)

    message = "\U0001F4A1 *Найперспективніші акції для купівлі:*"
    for _, row in df.iterrows():
        rec = escape_markdown(row['Recommendation'].capitalize())
        rsi = float(row['RSI'])
        change = float(row['Price Change %'])
        norm_change = row.get('Normalized Price Change %')
        direction = "🔼🟢" if change > 0 else "🔽🔴"
        if pd.isna(norm_change):
            norm_str = 'n/a'
        else:
            norm_dir = "🔼🟢" if norm_change > 0 else "🔽🔴"
            norm_str = f"{norm_dir} {abs(round(norm_change, 2))}%"

        rsi_flag = rsi < 40
        drop_flag = change <= -5
        rec_flag = row['Recommendation'] in ['buy', 'strong_buy']
        score = sum([rsi_flag, drop_flag, rec_flag])

        if score == 3:
            emoji = '🔥'
        elif score == 2:
            emoji = '✅'
        elif score == 1:
            emoji = '⚠️'
        else:
            emoji = '❌'

        risk_emoji = ''
        if rsi > 80:
            risk_emoji += " 🚫 Перегріта"
        elif rsi < 30:
            risk_emoji += " 🧊 Перепродана"
        elif rsi < 40:
            risk_emoji += " 📉 Потенціал"
        elif 40 <= rsi <= 70:
            risk_emoji += " ⚖️ Нейтральна"
        elif rsi > 70:
            risk_emoji += " 🔺 Оптимізм"

        if change > 30:
            risk_emoji += " ⚡ Висока волатильність"
        elif abs(change) < 2:
            risk_emoji += " 💧 Стабільна"

        ticker = escape_markdown(row['Ticker'])
        name = escape_markdown(row['Company'])
        pot = round(row['Potential %'], 2)
        msg_line = (
            f"\n{emoji} *{ticker}* {name}: ${row['Current Price']} | "
            f"Зміна: {direction} {abs(change)}% (норм: {norm_str}) | "
            f"RSI: {row['RSI']}{risk_emoji} | Реком: {rec}"
        )
        if row['Target Mean Price'] is not None:
            msg_line += f" | 🎯 ${row['Target Mean Price']}"
        msg_line += f" | Потенціал: +{pot}%"
        message += msg_line
    return message


def analyze_sell_signals(row):
    """Return sell score based on overbought conditions."""
    rsi_flag = float(row["RSI"]) > SELL_RSI_THRESHOLD
    growth_flag = float(row["Price Change %"]) > SELL_GROWTH_THRESHOLD
    rec_flag = str(row["Recommendation"]).lower() in SELL_RECOMMENDATION_KEYS
    return int(rsi_flag) + int(growth_flag) + int(rec_flag)


def sell_recommendations(df, limit=5):
    df = df.copy()
    if "Recommendation" in df.columns:
        df["Recommendation"] = df["Recommendation"].fillna("n/a").astype(str)

    df = df[df["Ticker"].isin(SELL_TICKERS)]
    df = df[df["Error"].isnull()]
    df["Sell Score"] = df.apply(analyze_sell_signals, axis=1)
    df = df[df["Sell Score"] >= SELL_SCORE_THRESHOLD]
    df = df.sort_values(by=["Sell Score", "Price Change %"], ascending=[False, False]).head(limit)

    message = "\U0001F4E4 *Рекомендовані до продажу:*"
    for _, row in df.iterrows():
        rec = escape_markdown(row["Recommendation"].capitalize())
        rsi = float(row["RSI"])
        change = float(row["Price Change %"])
        norm_change = row.get("Normalized Price Change %")
        direction = "🔼🟢" if change > 0 else "🔽🔴"
        if pd.isna(norm_change):
            norm_str = 'n/a'
        else:
            norm_dir = "🔼🟢" if norm_change > 0 else "🔽🔴"
            norm_str = f"{norm_dir} {abs(round(norm_change, 2))}%"
        rsi_mark = " 🚫" if rsi > SELL_RSI_THRESHOLD else ""

        ticker = escape_markdown(row["Ticker"])
        name = escape_markdown(row["Company"]) if row.get("Company") else ""
        msg_line = (
            f"\n- *{ticker}* {name} | Ціна: ${row['Current Price']} | "
            f"RSI: {row['RSI']}{rsi_mark} | Зміна: {direction} {abs(change)}% "
            f"(норм: {norm_str}) | Реком: {rec}"
        )
        if row.get("Target Mean Price") is not None:
            msg_line += f" | 🎯 ${row['Target Mean Price']}"
        msg_line += f" | Оцінка: {int(row['Sell Score'])}/3"
        message += msg_line
    return message

def collect_data():
    """Collect fresh stock data and save it to today's CSV file."""
    today = datetime.now().strftime('%Y-%m-%d')
    output_dir = os.path.join(os.getcwd(), RESULT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    results = [analyze_stock(ticker) for ticker in TICKERS]
    df = pd.DataFrame(results)

    output_path = os.path.join(output_dir, f"{today}.csv")
    df.to_csv(output_path, index=False)
    return df


def load_latest_csv():
    """Load the most recent CSV from the result directory."""
    result_path = os.path.join(os.getcwd(), RESULT_DIR)
    if not os.path.isdir(result_path):
        raise FileNotFoundError("No result directory found")

    files = sorted(f for f in os.listdir(result_path) if f.endswith(".csv"))
    if not files:
        raise FileNotFoundError("No CSV files found in result directory")

    latest = files[-1]
    df = pd.read_csv(os.path.join(result_path, latest))
    if "Recommendation" in df.columns:
        df["Recommendation"] = df["Recommendation"].fillna("n/a").astype(str)
    return df


def load_or_collect(date: Optional[str] = None):
    """Load data for the given date or collect today's data.

    When no date is provided or the provided date equals today, the function
    searches for CSV files for today and the two preceding days. This handles
    cases when the market is closed (e.g. weekends) or data collection was
    skipped. Data is only collected automatically on weekdays if no file exists
    for today or the previous days.
    """

    if date is None:
        base_date = datetime.now().date()
    else:
        base_date = datetime.strptime(date, "%Y-%m-%d").date()

    search_dates = [base_date]

    # If requesting today's data (either explicitly or implicitly), also check
    # the two previous days in case the market was closed or data wasn't
    # collected.
    if date is None or base_date == datetime.now().date():
        search_dates.append(base_date - timedelta(days=1))
        search_dates.append(base_date - timedelta(days=2))

    df = None
    found_date = None
    for d in search_dates:
        d_str = d.strftime("%Y-%m-%d")
        output_path = os.path.join(os.getcwd(), RESULT_DIR, f"{d_str}.csv")
        if os.path.exists(output_path):
            df = pd.read_csv(output_path)
            found_date = d_str
            break

    if df is None:
        # Collect data only on weekdays when no file was found
        if date is None and base_date.weekday() < 5:
            df = collect_data()
        else:
            raise FileNotFoundError(f"No data for {date or base_date.strftime('%Y-%m-%d')}")
    else:
        date = found_date

    # Ensure recommendation values are strings
    if 'Recommendation' in df.columns:
        df['Recommendation'] = df['Recommendation'].fillna('n/a').astype(str)

    return df


def report_console(date: Optional[str] = None):
    df = load_or_collect(date)
    for msg in format_sector_reports(df):
        print(msg)


def report_telegram(date: Optional[str] = None):
    df = load_or_collect(date)
    for msg in format_sector_reports(df):
        send_telegram_message(msg)


def offer_console(date: Optional[str] = None):
    df = load_or_collect(date)
    print(top_recommendations(df))


def offer_telegram(date: Optional[str] = None):
    df = load_or_collect(date)
    send_telegram_message(top_recommendations(df))


def offer_history(ticker: str):
    ticker = ticker.upper()
    result_path = os.path.join(os.getcwd(), RESULT_DIR)
    if not os.path.isdir(result_path):
        print("No result directory found")
        return

    files = sorted(f for f in os.listdir(result_path) if f.endswith(".csv"))
    for fname in files:
        date = fname[:-4]
        fpath = os.path.join(result_path, fname)
        df = pd.read_csv(fpath)
        if 'Recommendation' in df.columns:
            df['Recommendation'] = df['Recommendation'].fillna('n/a').astype(str)

        df = df[df['Error'].isnull() & df['Target Mean Price'].notnull()]
        df['Score'] = (
            (df['RSI'] < 40).astype(int) +
            (df['Price Change %'] <= -5).astype(int) +
            (df['Recommendation'].isin(['buy', 'strong_buy'])).astype(int)
        )
        df = df[df['Score'] >= 2]
        df['Potential %'] = (
            (df['Target Mean Price'] - df['Current Price']) / df['Current Price']
        ) * 100
        df = df.sort_values(by=['Score', 'Potential %'], ascending=[False, False]).head(5)
        match = df[df['Ticker'] == ticker]
        if not match.empty:
            row = match.iloc[0]
            rec = escape_markdown(row['Recommendation'].capitalize())
            rsi = float(row['RSI'])
            change = float(row['Price Change %'])
            norm_change = row.get('Normalized Price Change %')
            direction = "🔼🟢" if change > 0 else "🔽🔴"
            if pd.isna(norm_change):
                norm_str = 'n/a'
            else:
                norm_dir = "🔼🟢" if norm_change > 0 else "🔽🔴"
                norm_str = f"{norm_dir} {abs(round(norm_change, 2))}%"

            rsi_flag = rsi < 40
            drop_flag = change <= -5
            rec_flag = row['Recommendation'] in ['buy', 'strong_buy']
            score = sum([rsi_flag, drop_flag, rec_flag])

            if score == 3:
                emoji = '🔥'
            elif score == 2:
                emoji = '✅'
            elif score == 1:
                emoji = '⚠️'
            else:
                emoji = '❌'

            risk_emoji = ''
            if rsi > 80:
                risk_emoji += " 🚫 Перегріта"
            elif rsi < 30:
                risk_emoji += " 🧊 Перепродана"
            elif rsi < 40:
                risk_emoji += " 📉 Потенціал"
            elif 40 <= rsi <= 70:
                risk_emoji += " ⚖️ Нейтральна"
            elif rsi > 70:
                risk_emoji += " 🔺 Оптимізм"

            if change > 30:
                risk_emoji += " ⚡ Висока волатильність"
            elif abs(change) < 2:
                risk_emoji += " 💧 Стабільна"

            pot = round(row['Potential %'], 2)
            name = escape_markdown(row['Company']) if row['Company'] else ''
            ticker_md = escape_markdown(row['Ticker'])
            price = row['Target Mean Price']
            print(
                f"{date}: {emoji} *{ticker_md}* {name}: ${row['Current Price']} | "
                f"Зміна: {direction} {abs(change)}% (норм: {norm_str}) | "
                f"RSI: {row['RSI']}{risk_emoji} | Реком: {rec} | 🎯 ${price} | Потенціал: +{pot}%"
            )


def sell_console():
    df = load_latest_csv()
    print(sell_recommendations(df))


def sell_telegram():
    df = load_latest_csv()
    send_telegram_message(sell_recommendations(df))


def show_history(ticker: str):
    ticker = ticker.upper()
    result_path = os.path.join(os.getcwd(), RESULT_DIR)
    if not os.path.isdir(result_path):
        print("No result directory found")
        return

    files = sorted(f for f in os.listdir(result_path) if f.endswith(".csv"))
    rows = []
    for fname in files:
        date = fname[:-4]
        df = pd.read_csv(os.path.join(result_path, fname))
        match = df[df["Ticker"] == ticker]
        if not match.empty:
            row = match.iloc[0].copy()
            row["Date"] = date
            rows.append(row)

    if not rows:
        print("No history for", ticker)
        return

    hist_df = pd.DataFrame(rows)
    cols = ["Date"] + [c for c in hist_df.columns if c != "Date"]
    hist_df = hist_df[cols]
    print(hist_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Stock analysis utility")
    parser.add_argument(
        "command",
        choices=[
            "collect",
            "report_console",
            "report_telegram",
            "offer_console",
            "offer_telegram",
            "offer_history",
            "sell_console",
            "sell_telegram",
            "show_history",
        ],
        help="Action to perform",
    )
    parser.add_argument(
        "date",
        nargs="?",
        help="Date in YYYY-MM-DD format to load data from",
    )
    args = parser.parse_args()

    if args.command == "collect":
        collect_data()
    elif args.command == "report_console":
        report_console(args.date)
    elif args.command == "report_telegram":
        report_telegram(args.date)
    elif args.command == "offer_console":
        offer_console(args.date)
    elif args.command == "offer_telegram":
        offer_telegram(args.date)
    elif args.command == "offer_history":
        if not args.date:
            print("Ticker symbol required")
        else:
            offer_history(args.date)
    elif args.command == "sell_console":
        sell_console()
    elif args.command == "sell_telegram":
        sell_telegram()
    elif args.command == "show_history":
        if not args.date:
            print("Ticker symbol required")
        else:
            show_history(args.date)

if __name__ == "__main__":
    main()
