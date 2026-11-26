import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from pathlib import Path

# STEP 2. ETF 수익률 데이터 불러오기
returns = pd.read_csv(
    "data/etf_returns.csv",
    index_col=0,
    parse_dates=True
)

# 날짜순 정렬
returns = returns.sort_index()

# 데이터 확인
print("ETF 목록:", returns.columns.tolist())
print("전체 거래일 수:", len(returns))
print("데이터 시작일:", returns.index.min())
print("데이터 종료일:", returns.index.max())

# STEP 3. 백테스팅 기본 설정
lookback = 252
n_assets = len(returns.columns)

# 동일가중 투자 비중
equal_weights = np.ones(n_assets) / n_assets

# 거래비용: 거래금액의 0.1%
transaction_cost = 0.001

print("\n=== 연구 기본 설정 ===")
print("학습 기간:", lookback, "거래일")
print("ETF 개수:", n_assets)
print("거래비용:", transaction_cost * 100, "%")

# STEP 4. 공분산 행렬을 이용한 최소분산 최적화
def optimize_minvar(cov_matrix):

    # 목적함수: 포트폴리오 분산
    def objective(w):
        return w @ cov_matrix @ w

    # 투자 비중 합계 = 100%
    constraints = {
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1
    }

    # 공매도 금지: 각 비중 0~100%
    bounds = [(0, 1)] * n_assets

    result = minimize(
        objective,
        x0=equal_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000}
    )

    if not result.success:
        raise RuntimeError(result.message)

    return result.x

# STEP 5. 공분산 추정 방법 선택
def estimate_covariance(train_data, method):

    if method == "sample":
        return train_data.cov().to_numpy() * 252

    elif method == "ledoit_wolf":
        lw = LedoitWolf()
        lw.fit(train_data.to_numpy())
        return lw.covariance_ * 252

    else:
        raise ValueError("알 수 없는 공분산 추정 방법입니다.")

    # STEP 6. 첫 252거래일로 두 모델 테스트
train_data = returns.iloc[:lookback]

sample_cov = estimate_covariance(train_data, "sample")
lw_cov = estimate_covariance(train_data, "ledoit_wolf")

sample_weights = optimize_minvar(sample_cov)
lw_weights = optimize_minvar(lw_cov)

print("\n=== 두 최소분산 모델 비교 ===")

for i, ticker in enumerate(returns.columns):
    print(
        ticker,
        "| Sample:", round(sample_weights[i] * 100, 2), "%",
        "| Ledoit-Wolf:", round(lw_weights[i] * 100, 2), "%"
    )

print("Sample 비중 합계:", round(sample_weights.sum(), 6))
print("Ledoit-Wolf 비중 합계:", round(lw_weights.sum(), 6))

# STEP 7. 월별 리밸런싱 날짜 생성
eligible_returns = returns.iloc[lookback:]

monthly_starts = eligible_returns.groupby(
    eligible_returns.index.to_period("M")
).head(1)

rebalance_dates = monthly_starts.index

print("\n=== 전체 백테스팅 설정 ===")
print("첫 리밸런싱 날짜:", rebalance_dates[0])
print("마지막 리밸런싱 날짜:", rebalance_dates[-1])
print("총 리밸런싱 횟수:", len(rebalance_dates))

# STEP 8. 전체 기간 Walk-forward 백테스팅

strategies = ["Equal Weight", "Sample MinVar", "Ledoit-Wolf MinVar"]

# 각 전략의 보유 자산 가치
holdings = {name: np.zeros(n_assets) for name in strategies}

# 결과 저장용
daily_results = []
weight_records = []
turnover_records = []

# 리밸런싱 날짜를 빠르게 확인하기 위한 집합
rebalance_set = set(rebalance_dates)

for i in range(lookback, len(returns)):

    date = returns.index[i]
    daily_asset_returns = returns.iloc[i].to_numpy()

    # 매월 첫 거래일에만 투자 비중 재계산
    if date in rebalance_set:

        # 리밸런싱 당일은 제외하고 직전 252거래일 선택
        train = returns.iloc[i - lookback:i]

        sample_cov = estimate_covariance(train, "sample")
        lw_cov = estimate_covariance(train, "ledoit_wolf")

        target_weights = {
            "Equal Weight": equal_weights,
            "Sample MinVar": optimize_minvar(sample_cov),
            "Ledoit-Wolf MinVar": optimize_minvar(lw_cov)
        }

        for name in strategies:

            old_value = holdings[name].sum()

            # 현재 보유 비중과 새로운 목표 비중 비교
            if old_value == 0:
                current_weights = np.zeros(n_assets)
                old_value = 1.0
            else:
                current_weights = holdings[name] / old_value

            target = target_weights[name]

            # 거래량 및 거래비용 계산
            turnover = np.abs(target - current_weights).sum()
            cost = turnover * transaction_cost

            # 거래비용 차감 후 새 투자 비중 적용
            holdings[name] = old_value * (1 - cost) * target

            weight_records.append({
                "Date": date,
                "Strategy": name,
                **dict(zip(returns.columns, target))
            })

            turnover_records.append({
                "Date": date,
                "Strategy": name,
                "Turnover": turnover
            })

    # 당일 실제 ETF 수익률 반영
    row = {"Date": date}

    for name in strategies:

        before = holdings[name].sum()

        holdings[name] *= (1 + daily_asset_returns)

        after = holdings[name].sum()

        # 누적 자산가치 저장
        row[name] = after

    daily_results.append(row)

# 데이터프레임으로 정리
wealth = pd.DataFrame(daily_results).set_index("Date")

monthly_weights = pd.DataFrame(weight_records)
turnover_data = pd.DataFrame(turnover_records)

# 분석 결과 확인
print("\n=== 전체 백테스팅 결과 ===")
print("검증 거래일:", len(wealth))
print("리밸런싱 횟수:", len(rebalance_dates))
print("\n최종 자산가치:")
print(wealth.iloc[-1].round(4))

# STEP 9. 전체 표본 외 성과지표 계산

# 최초 투자금 1을 기준으로 일별 수익률 계산
daily_returns = wealth.pct_change()
daily_returns.iloc[0] = wealth.iloc[0] - 1

# CAGR: 연평균 복리수익률
cagr = wealth.iloc[-1] ** (252 / len(wealth)) - 1

# 연율화 변동성
volatility = daily_returns.std() * np.sqrt(252)

# 샤프지수: 무위험수익률 0% 가정
sharpe = daily_returns.mean() * 252 / volatility

# 최대낙폭: 최초 자산가치 1도 포함
peak = wealth.cummax().clip(lower=1.0)
drawdown = wealth / peak - 1
mdd = drawdown.min()

# 성과 비교표
performance = pd.DataFrame({
    "CAGR (%)": cagr * 100,
    "Volatility (%)": volatility * 100,
    "Sharpe": sharpe,
    "MDD (%)": mdd * 100
})

print("\n=== 전체 표본 외 성과 비교 ===")
print(performance.round(2).to_string())

# STEP 10. 리밸런싱 거래량 분석

# 첫 매수는 제외하고 리밸런싱 거래량만 분석
rebalance_turnover = turnover_data[
    turnover_data["Date"] != rebalance_dates[0]
]

avg_turnover = rebalance_turnover.groupby(
    "Strategy"
)["Turnover"].mean() * 100

print("\n=== 평균 월별 거래량 ===")
print(avg_turnover.round(2).to_string())

# 매월 목표 투자 비중의 변화량 계산
weight_changes = {}

for name in strategies:
    w = monthly_weights[
        monthly_weights["Strategy"] == name
    ].sort_values("Date")

    changes = w[returns.columns].diff().abs().sum(axis=1)

    # 최초 비중 설정은 제외
    weight_changes[name] = changes.iloc[1:].mean() * 100

print("\n=== 평균 월별 목표 비중 변화량 ===")

for name, value in weight_changes.items():
    print(name, ":", round(value, 2), "%p")

print(performance.round(2).to_string())

# STEP 10. 리밸런싱 거래량 분석

# 첫 매수는 제외하고 리밸런싱 거래량만 분석
rebalance_turnover = turnover_data[
    turnover_data["Date"] != rebalance_dates[0]
]

avg_turnover = rebalance_turnover.groupby(
    "Strategy"
)["Turnover"].mean() * 100

print("\n=== 평균 월별 거래량 ===")
print(avg_turnover.round(2).to_string())

# 매월 목표 투자 비중의 변화량 계산
weight_changes = {}

for name in strategies:
    w = monthly_weights[
        monthly_weights["Strategy"] == name
    ].sort_values("Date")

    changes = w[returns.columns].diff().abs().sum(axis=1)

    # 최초 비중 설정은 제외
    weight_changes[name] = changes.iloc[1:].mean() * 100

print("\n=== 평균 월별 목표 비중 변화량 ===")

for name, value in weight_changes.items():
    print(name, ":", round(value, 2), "%p")

# STEP 11. 학습기간 민감도 분석
lookback_list = [126, 252, 504]
sensitivity_results = []

# 세 학습기간이 동일한 미래 구간에서 평가되도록 설정
common_start = 504
common_returns = returns.iloc[common_start:]

common_dates = common_returns.groupby(
    common_returns.index.to_period("M")
).head(1).index

common_rebalance_set = set(common_dates)

for window in lookback_list:

    # 각 전략별 초기 보유 자산
    portfolios = {
        name: np.zeros(n_assets) for name in strategies
    }

    history = []

    for i in range(common_start, len(returns)):

        date = returns.index[i]
        asset_returns = returns.iloc[i].to_numpy()

        if date in common_rebalance_set:

            # 리밸런싱 직전 window개 거래일 사용
            train = returns.iloc[i-window:i]

            sample_cov = estimate_covariance(train, "sample")
            lw_cov = estimate_covariance(train, "ledoit_wolf")

            targets = {
                "Equal Weight": equal_weights,
                "Sample MinVar": optimize_minvar(sample_cov),
                "Ledoit-Wolf MinVar": optimize_minvar(lw_cov)
            }

            for name in strategies:

                old_value = portfolios[name].sum()

                if old_value == 0:
                    old_value = 1.0
                    current_weights = np.zeros(n_assets)
                else:
                    current_weights = portfolios[name] / old_value

                turnover = np.abs(
                    targets[name] - current_weights
                ).sum()

                cost = turnover * transaction_cost

                portfolios[name] = (
                    old_value * (1 - cost) * targets[name]
                )

        # 리밸런싱 여부와 관계없이 일별 수익률 반영
        row = {"Date": date}

        for name in strategies:
            portfolios[name] *= (1 + asset_returns)
            row[name] = portfolios[name].sum()

        history.append(row)

    # 해당 학습기간의 누적 자산가치
    window_wealth = pd.DataFrame(history).set_index("Date")

    # 일별 수익률 계산
    window_returns = window_wealth.pct_change()
    window_returns.iloc[0] = window_wealth.iloc[0] - 1

    # 성과지표 계산
    for name in strategies:

        cagr_value = (
            window_wealth[name].iloc[-1]
            ** (252 / len(window_wealth)) - 1
        )

        vol_value = (
            window_returns[name].std() * np.sqrt(252)
        )

        sensitivity_results.append({
            "Lookback": window,
            "Strategy": name,
            "CAGR (%)": cagr_value * 100,
            "Volatility (%)": vol_value * 100
        })

# 결과표 만들기
sensitivity = pd.DataFrame(sensitivity_results)

print("\n=== 학습기간 민감도 분석 ===")
print(sensitivity.round(2).to_string(index=False))

#step12. 거래비용 민감도 분석

cost_list = [0, 0.001, 0.0025]
cost_results =[]

# STEP 12. 거래비용 민감도 분석

cost_list = [0, 0.001, 0.0025]
cost_results = []

for cost_rate in cost_list:

    # 전략별 초기 보유 자산가치
    portfolios = {
        name: np.zeros(n_assets) for name in strategies
    }

    # 월별 목표 비중을 날짜별로 정리
    target_lookup = {}

    for name in strategies:
        selected = monthly_weights[
            monthly_weights["Strategy"] == name
        ]

        target_lookup[name] = {
            row["Date"]: row[returns.columns].to_numpy(dtype=float)
            for _, row in selected.iterrows()
        }

    # 전체 미래 기간에 동일한 투자 비중 적용
    for i in range(lookback, len(returns)):

        date = returns.index[i]
        asset_returns = returns.iloc[i].to_numpy()

        for name in strategies:

            # 리밸런싱 날짜에만 거래
            if date in target_lookup[name]:

                old_value = portfolios[name].sum()

                if old_value == 0:
                    old_value = 1.0
                    current_weights = np.zeros(n_assets)
                else:
                    current_weights = portfolios[name] / old_value

                target = target_lookup[name][date]

                turnover = np.abs(
                    target - current_weights
                ).sum()

                cost_amount = turnover * cost_rate

                portfolios[name] = (
                    old_value * (1 - cost_amount) * target
                )

            # 당일 ETF 수익률 반영
            portfolios[name] *= (1 + asset_returns)

    # 최종 성과 저장
    for name in strategies:

        final_wealth = portfolios[name].sum()

        cagr_value = (
            final_wealth ** (252 / (len(returns) - lookback))
            - 1
        )

        cost_results.append({
            "Transaction Cost (bp)": int(cost_rate * 10000),
            "Strategy": name,
            "Final Wealth": final_wealth,
            "CAGR (%)": cagr_value * 100
        })

# 결과표 생성
cost_sensitivity = pd.DataFrame(cost_results)

print("\n=== 거래비용 민감도 분석 ===")
print(cost_sensitivity.round(4).to_string(index=False))

# STEP 13. 연구 결과 저장

output_dir = Path("results/full_research")
output_dir.mkdir(parents=True, exist_ok=True)

performance.to_csv(output_dir / "performance_metrics.csv")
sensitivity.to_csv(output_dir / "lookback_sensitivity.csv", index=False)
cost_sensitivity.to_csv(
    output_dir / "transaction_cost_sensitivity.csv",
    index=False
)
monthly_weights.to_csv(output_dir / "monthly_weights.csv", index=False)

print("\nCSV 분석 결과 저장 완료!")

# 누적 자산가치 그래프
wealth.plot(figsize=(10,6))
plt.title("Out-of-Sample Cumulative Wealth")
plt.xlabel("Date")
plt.ylabel("Portfolio Wealth (Initial = 1)")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(output_dir / "cumulative_wealth.png", dpi=300)
plt.close()

# 표본 외 변동성 비교
performance["Volatility (%)"].plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("Out-of-Sample Volatility")
plt.ylabel("Annualized Volatility (%)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(output_dir / "volatility_comparison.png", dpi=300)
plt.close()

# 표본 외 변동성 비교
performance["Volatility (%)"].plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("Out-of-Sample Volatility")
plt.ylabel("Annualized Volatility (%)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(output_dir / "volatility_comparison.png", dpi=300)
plt.close()

# Sample과 Ledoit-Wolf 투자 비중 변화
for name, filename in [
    ("Sample MinVar", "sample_weights.png"),
    ("Ledoit-Wolf MinVar", "lw_weights.png")
]:
    selected = monthly_weights[
        monthly_weights["Strategy"] == name
    ].set_index("Date")

    selected[returns.columns].plot(
        kind="area",
        stacked=True,
        figsize=(11, 6)
    )

    plt.title(name + " - Portfolio Weights Over Time")
    plt.ylabel("Portfolio Weight")
    plt.xlabel("Date")
    plt.ylim(0, 1)
    plt.tight_layout()

    plt.savefig(output_dir / filename, dpi=300)
    plt.close()

    # 학습기간에 따른 표본 외 변동성 비교
lookback_chart = sensitivity.pivot(
    index="Lookback",
    columns="Strategy",
    values="Volatility (%)"
)

lookback_chart.plot(
    marker="o",
    figsize=(9, 5)
)

plt.title("Lookback Sensitivity - Volatility")
plt.xlabel("Lookback Window (Trading Days)")
plt.ylabel("Annualized Volatility (%)")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(output_dir / "lookback_sensitivity.png", dpi=300)
plt.close()

print("그래프 저장 완료!")

#step14기본 검증 테스트

#1. 모든 목표 투자 비중 합계가 100%인지 확인
weight_sums = monthly_weights[returns.columns].sum(axis=1)
assert np.allclose(weight_sums, 1.0)

# 2. 투자 비중에 음수가 없는지 확인
assert (monthly_weights[returns.columns] >= -1e-8).all().all()

# 3. 세 모델의 최종 자산가치가 양수인지 확인
assert (wealth.iloc[-1] > 0).all()

# 4. 거래비용 10bp 결과와 기본 백테스트 비교
base_cost = cost_sensitivity[
    cost_sensitivity["Transaction Cost (bp)"] == 10
].set_index("Strategy")["Final Wealth"]

assert np.allclose(
    base_cost.loc[strategies].to_numpy(),
    wealth.iloc[-1][strategies].to_numpy()
)

print("\n기본 검증 4개 통과!")

# STEP 14-2. 미래 정보 사용 여부 검증

for date in rebalance_dates:

    i = returns.index.get_loc(date)

    # 리밸런싱 직전 252거래일 선택
    train = returns.iloc[i - lookback:i]

    # 학습 데이터가 정확히 252개인지 확인
    assert len(train) == lookback

    # 학습 종료일이 리밸런싱 날짜보다 이전인지 확인
    assert train.index.max() < date

    # 리밸런싱 날짜가 학습 데이터에 없는지 확인
    assert date not in train.index

print("미래 정보 사용 여부 검증 통과!")

# STEP 14-3. 거래비용 검증

for name in strategies:

    selected = cost_sensitivity[
        cost_sensitivity["Strategy"] == name
    ].sort_values("Transaction Cost (bp)")

    values = selected["Final Wealth"].to_numpy()

    # 거래비용이 높아지면 최종 자산가치가 증가하지 않아야 함
    assert np.all(np.diff(values) <= 1e-8)

print("거래비용 민감도 검증 통과!")

# STEP 14-4. 성과지표 검산

# 최종 자산가치에서 CAGR 독립 재계산
check_cagr = (
    wealth.iloc[-1] ** (252 / len(wealth)) - 1
) * 100

# 기존 결과와 비교
assert np.allclose(
    check_cagr.loc[strategies].to_numpy(),
    performance.loc[strategies, "CAGR (%)"].to_numpy()
)

# 일별 수익률을 이용한 연율화 변동성 재계산
check_volatility = daily_returns.std() * np.sqrt(252) * 100

assert np.allclose(
    check_volatility.loc[strategies].to_numpy(),
    performance.loc[strategies, "Volatility (%)"].to_numpy()
)

print("성과지표 검산 통과!")