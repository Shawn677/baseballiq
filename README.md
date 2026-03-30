# BaseballIQ

A baseball analytics platform with a live Streamlit dashboard and weekly AI-written newsletter.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in your API keys:
   ```bash
   cp .env.example .env
   ```

## Running

```bash
# Run the dashboard
streamlit run dashboard/app.py

# Generate and send the newsletter manually
python newsletter/generator.py
```

## Project Structure

```
baseballiq/
├── data/
│   ├── fetcher.py       # All MLB Stats API calls
│   ├── processor.py     # Data cleaning and stat calculations
│   └── cache/           # Cached JSON responses (auto-created)
├── dashboard/
│   └── app.py           # Streamlit dashboard
├── newsletter/
│   ├── generator.py     # Claude API newsletter writer
│   └── sender.py        # Beehiiv email sender
├── requirements.txt
└── README.md
```

## Data Sources

- **MLB Stats API** — free, no key required
- **pybaseball** — wraps Baseball Reference / FanGraphs scraping, free, no key required

## Environment Variables

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Your Anthropic API key (for newsletter generation) |
| `BEEHIIV_API_KEY` | Your Beehiiv API key (for email sending) |
| `BEEHIIV_PUBLICATION_ID` | Your Beehiiv publication ID |
