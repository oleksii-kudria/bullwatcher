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
python main.py report_console    # print sector report from latest data
python main.py report_telegram   # send sector report to Telegram
python main.py offer_console     # print top recommendations
python main.py offer_telegram    # send top recommendations to Telegram
```

Run `collect` once per day before other commands, or rely on the report/offer
commands to automatically load the most recent CSV in the `result` directory.
