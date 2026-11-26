import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize

returns = pd.read_csv(
    "data/etf_returns.csv",
    index_col=0,
    parse_dates=True
)

cov_matrix = returns.cov() * 252
n_assets = len(returns.columns)

weights = np.ones(n_assets) / n_assets

print("투자 비중:", weights)
print("비중 합계:", weights.sum())

#동일가중 포트폴리오 분산 계산 
equal_variance = weights @ cov_matrix @ weights

# 분산의 제곱근 = 변동성
portfolio_volatility = np.sqrt(equal_variance)

print("포트폴리오 분산:",round(equal_variance, 4))
print("동일가중 포트폴리오 연율화 변동성:", round(portfolio_volatility * 100, 2), "%")

#투자 비중을 입력받아 포트폴리오 분산 계산
def portfolio_variance(weights):
    variance = weights @ cov_matrix @ weights
    return variance 

portfolio_variance(weights)
# 함수가 제대로 작동하는지 확인
test_variance = portfolio_variance(weights)

print("함수로 계산한 분산:", round(test_variance, 4))

#투자 비중 합계가 1이 되어야함 
constraints = {"type": "eq", "fun" : lambda w : np.sum(w) - 1}

# 각 ETF의 투자 비중은 0~1 사이 
bounds = [(0, 1)] * n_assets

# 최소분산 포트폴리오 최적화 실행
result = minimize(
    portfolio_variance,
    x0=weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"ftol": 1e-12, "maxiter": 1000}
)

# 최적화 성공 여부 확인
print("최적화 성공 여부:", result.success)
print("최적화 메시지:", result.message)

if not result.success:
    raise RuntimeError("최적화에 실패했습니다.")

# 최적 투자 비중 추출
optimal_weights = result.x

# 최적 포트폴리오 변동성 계산
optimal_variance = portfolio_variance(optimal_weights)
optimal_volatility = np.sqrt(optimal_variance)

print("\n=== 최소분산 포트폴리오 ===")

for ticker, weight in zip(returns.columns, optimal_weights):
    print(ticker, ":", round(weight * 100, 2), "%")

print("투자 비중 합계:", round(optimal_weights.sum(), 6))
print("최소분산 포트폴리오 연율화 변동성:", round(optimal_volatility * 100, 2), "%")

#두 포트폴리오의 투자 비중 비교표
comparison = pd. DataFrame({
    "Equal Weight": weights,
    "Minimum Variance": optimal_weights
    }, index=returns.columns)

print("\n===투자 비중 비교 ===")
print((comparison * 100).round(2))

# 비교표를 CSV 파일로 저장
comparison.to_csv("results/portfolio_weights.csv")

print("투자 비중 비교표 저장 완료!")

#첫번째 그래프 - ETF별 투자 비중 비교 그래프
comparison.plot(kind="bar", figsize=(10, 5))

plt.title("Portfolio Weight Comparison")
plt.ylabel("Portfolio Weight")
plt.xlabel("ETF")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("results/portfolio_weights.png", dpi=300)
plt.close()

#포트폴리오별 연율화 변동성 비교
volatility_comparison = pd.Series({
    "Equal Weight": portfolio_volatility * 100,
    "Minimum Variance": optimal_volatility * 100
})
volatility_comparison.plot(kind="bar", figsize=(8,5))
plt.title("Annualized Volatility Comparison")
plt.ylabel("Annualized Volatility (%)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("results/volatility_comparison.png", dpi=300)
plt.close()

print("그래프 2개 저장 완료!")