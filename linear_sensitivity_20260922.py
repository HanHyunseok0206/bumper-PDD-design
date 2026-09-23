"""
linear_sensitivity_20260922.py

목적: design_matrix_diagnostics_20260922.py에서 확인했듯 입력변수들은 서로 독립적으로 흔들렸지만
      (다중공선성 없음, PCA 유효차원 15/19) 변동폭이 공칭값 대비 1.7~3.3%로 매우 좁음. 이런 데이터에서
      비선형+상호작용 항을 같이 추정하는 PDD(19변수, 표본 90)는 파라미터 대비 표본이 부족해 불안정했음
      (pdd_first_pass_20260922.py, 10-seed 검증 R2가 -0.50~+0.36으로 흔들림).
      선형 항만 쓰는 표준화 회귀는 추정할 파라미터가 훨씬 적어(19+절편) 같은 표본으로도 더 안정적인
      민감도 순위를 줄 것으로 기대하고 재확인.

방법:
  - 표준화 회귀계수(beta) = 원단위 기울기 * (X 표준편차 / Y 표준편차) -> 변수 중요도 순위
  - 학습 R^2만 보지 않고 5-fold CV R^2 확인, 10개 시드로 반복해 안정성 체크 (PDD 쪽과 동일 취지,
    CLAUDE.md: 학습 R^2만으로 판단 금지)
  - 부트스트랩 재표본 200회로 각 변수가 |beta| 상위 5위 안에 드는 빈도 확인 -> 순위 자체의 강건성 정량화
"""

import numpy as np
from pdd_first_pass_20260922 import load_clean_data, VAR_COLS


def ols_fit(X, Y):
    n = X.shape[0]
    A = np.column_stack([np.ones(n), X])
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    return coef  # coef[0]=절편, coef[1:]=원단위 기울기


def standardized_coefs(X, Y, coef_raw):
    sx = X.std(axis=0, ddof=1)
    sy = Y.std(ddof=1)
    return coef_raw[1:] * sx / sy


def r2_score(Y, pred):
    ss_res = np.sum((Y - pred) ** 2)
    ss_tot = np.sum((Y - Y.mean()) ** 2)
    return 1 - ss_res / ss_tot


def kfold_cv_r2(X, Y, k=5, seed=0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    r2s = []
    for i in range(k):
        val_idx = folds[i]
        tr_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        coef = ols_fit(X[tr_idx], Y[tr_idx])
        pred = coef[0] + X[val_idx] @ coef[1:]
        r2s.append(r2_score(Y[val_idx], pred))
    return np.array(r2s)


def bootstrap_top5_freq(X, Y, n_boot=200, top_k=5, seed=0):
    rng = np.random.default_rng(seed)
    n, p = X.shape
    count = np.zeros(p)
    for _ in range(n_boot):
        boot_idx = rng.integers(0, n, size=n)
        Xb, Yb = X[boot_idx], Y[boot_idx]
        if np.any(Xb.std(axis=0) == 0):
            continue
        coef_raw = ols_fit(Xb, Yb)
        std_coef = standardized_coefs(Xb, Yb, coef_raw)
        top_idx = np.argsort(-np.abs(std_coef))[:top_k]
        count[top_idx] += 1
    return count / n_boot


def main():
    df = load_clean_data("doe_results_raw_20260922.csv")
    X = df[VAR_COLS].to_numpy()
    Y = df["SEA"].to_numpy()

    coef_raw = ols_fit(X, Y)
    std_coef = standardized_coefs(X, Y, coef_raw)
    pred_full = coef_raw[0] + X @ coef_raw[1:]
    train_r2 = r2_score(Y, pred_full)
    print(f"전체 표본(90) 학습 R2(참고용, 과신 금지): {train_r2:.4f}")

    print("\n=== 표준화 회귀계수(beta) 기준 민감도 순위 ===")
    order = np.argsort(-np.abs(std_coef))
    for i in order:
        print(f"  {VAR_COLS[i]:5s} beta={std_coef[i]:+.4f}")

    r2s = kfold_cv_r2(X, Y, k=5, seed=0)
    print(f"\n5-fold CV R2 (seed=0): {[round(v, 3) for v in r2s]}")
    print(f"평균 {r2s.mean():.3f}, 표준편차 {r2s.std():.3f}")

    cv_means = [kfold_cv_r2(X, Y, k=5, seed=s).mean() for s in range(10)]
    print(f"\n10개 시드 평균 CV R2: {[round(v, 3) for v in cv_means]}")
    print(f"평균 {np.mean(cv_means):.3f}, 표준편차 {np.std(cv_means):.3f}"
          "  (PDD의 10-seed 결과와 직접 비교할 것)")

    freq = bootstrap_top5_freq(X, Y, n_boot=200, top_k=5, seed=0)
    print("\n=== 부트스트랩 200회: |beta| 상위 5위 안에 든 빈도 ===")
    order2 = np.argsort(-freq)
    for i in order2:
        print(f"  {VAR_COLS[i]:5s} {freq[i]*100:5.1f}%")


if __name__ == "__main__":
    main()
