
import yfinance as yf
import pandas as pd
from pathlib import Path

tickers = ["SPY", "QQQ", "IWM", "EFA", "IEF", "GLD"]

# ETF별 가격을 하나씩 다운로드
price_data = {}

for ticker in tickers:
    data = yf.download(
        ticker,
        start="2015-01-01",
        end="2026-11-14",
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise ValueError(f"{ticker} 데이터 다운로드 실패")

    price_data[ticker] = data["Close"].iloc[:, 0]

# 하나의 가격 데이터 표로 합치기
prices = pd.DataFrame(price_data)

# 날짜순 정렬 및 결측치 제거
prices = prices.sort_index().dropna()

if prices.empty:
    raise ValueError("가격 데이터가 비어 있습니다.")

# CSV로 저장
Path("data").mkdir(exist_ok=True)
prices.to_csv("data/etf_prices.csv", index_label="Date")

print("=== ETF 가격 데이터 저장 완료 ===")
print(prices.head())
print("데이터 행 개수:", len(prices))
print("ETF 개수:", len(prices.columns))
