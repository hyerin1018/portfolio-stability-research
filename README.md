# portfolio-stability-research
Empirical analysis of portfolio optimization stability

import numpy as np

# 두 자산의 예상 연간 수익률
returns = np.array([0.10, 0.05])

# 투자 비중
weights = np.array([0.6, 0.4])

# 포트폴리오 기대수익률 계산
portfolio_return = np.dot(weights, returns)

print("포트폴리오 기대수익률:", round(portfolio_return * 100, 2), "%")
