# main.py
import os
from datetime import datetime
import argparse
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
import requests
from config import (
    TICKERS,
    RSI_THRESHOLD,
    PRICE_DROP_THRESHOLD,
    RESULT_DIR,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
)
import numpy as np
import re

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
        current_price = df['Close'].iloc[-1].item()
        month_open_price = df['Close'].iloc[0].item()
        price_change = ((current_price - month_open_price) / month_open_price) * 100

        close_series = df['Close']
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.squeeze()

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
    df['Sector'] = df['Sector'].fillna('Unknown')
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
            msg_line = f"\n{emoji} *{ticker}* {name}: ${row['Current Price']} | Зміна: {direction} {abs(row['Price Change %'])}% | RSI: {row['RSI']}{risk_emoji} | Реком: {rec}"
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
        ticker = escape_markdown(row['Ticker'])
        name = escape_markdown(row['Company'])
        rec = escape_markdown(row['Recommendation'].capitalize())
        pot = round(row['Potential %'], 2)
        message += f"\n- *{ticker}* {name} | Потенціал: +{pot}% | Реком: {rec} | 🎯 ${row['Target Mean Price']}"
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


def load_or_collect():
    """Load today's data if available, otherwise collect it."""
    today = datetime.now().strftime('%Y-%m-%d')
    output_path = os.path.join(os.getcwd(), RESULT_DIR, f"{today}.csv")
    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
    else:
        df = collect_data()

    # Ensure recommendation values are strings
    if 'Recommendation' in df.columns:
        df['Recommendation'] = df['Recommendation'].fillna('n/a').astype(str)

    return df


def report_console():
    df = load_or_collect()
    for msg in format_sector_reports(df):
        print(msg)


def report_telegram():
    df = load_or_collect()
    for msg in format_sector_reports(df):
        send_telegram_message(msg)


def offer_console():
    df = load_or_collect()
    print(top_recommendations(df))


def offer_telegram():
    df = load_or_collect()
    send_telegram_message(top_recommendations(df))


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
        ],
        help="Action to perform",
    )
    args = parser.parse_args()

    if args.command == "collect":
        collect_data()
    elif args.command == "report_console":
        report_console()
    elif args.command == "report_telegram":
        report_telegram()
    elif args.command == "offer_console":
        offer_console()
    elif args.command == "offer_telegram":
        offer_telegram()

if __name__ == "__main__":
    main()
