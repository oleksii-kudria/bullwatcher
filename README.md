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
```

Run `collect` once per day before other commands, or rely on the report/offer
commands to automatically load the most recent CSV in the `result` directory.
