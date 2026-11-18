
import numpy as np

returns = np.array([0.10, 0.05])
weights = np.array([0.6, 0.4])

portfolio_return = np.dot(weights, returns)

print("포트폴리오 기대수익률:", round(portfolio_return * 100, 2), "%")
