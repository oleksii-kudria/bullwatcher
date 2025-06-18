# bullwatcher

Utility for collecting stock data and generating daily reports.

## Installation

Install dependencies using `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage

```
python main.py collect           # fetch data and save result/YYYY-MM-DD.csv
python main.py report_console [YYYY-MM-DD]  # print sector report, optionally from a specific date
python main.py report_telegram [YYYY-MM-DD] # send sector report to Telegram
python main.py offer_console [YYYY-MM-DD]   # print top recommendations
python main.py offer_telegram [YYYY-MM-DD]  # send top recommendations to Telegram
python main.py offer_history TICKER        # show dates when the ticker was recommended
python main.py sell_console                # show sell recommendations in console
python main.py sell_telegram               # send sell recommendations to Telegram
python main.py show_history TICKER         # display historical rows for ticker
```

Run `collect` once per day before other commands, or rely on the report/offer
commands to automatically load the most recent CSV in the `result` directory.
