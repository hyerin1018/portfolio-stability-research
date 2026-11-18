import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 1. 이미 계산한 일별 ETF 수익률 불러오기
returns = pd.read_csv(
    "data/etf_returns.csv",
    index_col=0,
    parse_dates=True
)

# 2. ETF간 상관계수 계산
corr_matrix = returns.corr()

print("=== ETF간 상관계수 행렬 ===")
print(corr_matrix.round(2))

# 3. 결과 저장 폴더 확인
Path("results").mkdir(exist_ok=True)

# 4. 히트맵 그리기
plt.figure(figsize=(8, 6))
img = plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)

# 색상 막대
plt.colorbar(img)

# 축 이름 넣기
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.yticks(range(len(corr_matrix.index)), corr_matrix.index)

# 각 칸에 숫자 넣기
for i in range(len(corr_matrix.index)):
    for j in range(len(corr_matrix.columns)):
        plt.text(
            j, i,
            f"{corr_matrix.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            color="black"
        )

plt.title("ETF Correlation Heatmap")
plt.tight_layout()

# 5. 이미지 저장
plt.savefig("results/correlation_heatmap.png", dpi=300)
plt.show()

print("상관계수 히트맵 저장 완료!")