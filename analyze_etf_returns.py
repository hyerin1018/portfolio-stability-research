
import pandas as pd
import numpy as np
from pathlib import Path

# 1. 실제 ETF 가격 데이터 불러오기
prices = pd.read_csv(
    "data/etf_prices.csv",
    index_col=0,
    parse_dates=True
)

# 2. 일별 수익률 계산
returns = prices.pct_change().dropna()

# 3. ETF별 평균 수익률 (연율화)
annual_returns = returns.mean() * 252

# 4. ETF별 변동성 (연율화)
annual_volatility = returns.std() * np.sqrt(252)

# 5. 공분산 행렬 (연율화)
cov_matrix = returns.cov() * 252

# 6. 결과 출력
print("=== ETF별 연율화 평균 수익률 ===")
print((annual_returns * 100).round(2))

print("\n=== ETF별 연율화 변동성 ===")
print((annual_volatility * 100).round(2))

print("\n=== ETF 공분산 행렬 ===")
print(cov_matrix.round(4))

# 7. 분석 결과 저장
Path("results").mkdir(exist_ok=True)

returns.to_csv("data/etf_returns.csv")
cov_matrix.to_csv("results/covariance_matrix.csv")

print("\n분석 결과 저장 완료!")
