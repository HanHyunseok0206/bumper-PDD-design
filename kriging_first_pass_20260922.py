"""
kriging_first_pass_20260922.py

목적: design_matrix_diagnostics_20260922.py(2번)·linear_sensitivity_20260922.py(3번) 결과로
      "표본(90) 대비 변수(19) 부족 + 범위(1.7~3.3%)가 좁아 신호가 노이즈에 묻힌다"는 가설을 세운 뒤,
      실제 최종 서로게이트 후보인 크리깅도 같은 데이터에서 같은 한계를 보이는지 확인.
      Kriging_bumper_SEA.ipynb 골격 2단계와 같은 커널(Matern nu=2.5 + WhiteKernel)을 쓰되,
      DESIGN_IDX/NOISE_IDX가 아직 정해지지 않아 19변수 전체로 학습.

      주의: 실제 LB/UB를 아직 팀원에게 확인 못 해서 표본 min/max로 스케일링함(GUIDELINE 원칙상
      이론적 정의역이 맞지만 미확인 상태 — 확인되면 Kriging_bumper_SEA.ipynb의 theoretical_scale로 교체).

방법:
  - 단일 train/test 분할(노트북 골격 방식) 대신 5-fold CV R2로 확인 — 표본이 적어 분할 하나로는
    PDD 때처럼 운 좋게/나쁘게 나올 수 있어 안 믿음
  - 10개 시드로 CV 반복해 PDD(-0.50~+0.36)/선형회귀(-0.065±0.073) 결과와 나란히 비교
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from pdd_first_pass_20260922 import load_clean_data, VAR_COLS, min_max_scale


def make_gpr(seed=0):
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(1e-3)
    return GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=5, random_state=seed
    )


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
        gpr = make_gpr(seed=seed)
        gpr.fit(X[tr_idx], Y[tr_idx])
        pred = gpr.predict(X[val_idx])
        r2s.append(r2_score(Y[val_idx], pred))
    return np.array(r2s)


def main():
    df = load_clean_data("doe_results_raw_20260922.csv")
    X_raw = df[VAR_COLS].to_numpy().T  # (dim, N), min_max_scale 관례 맞춤
    Y = df["SEA"].to_numpy()
    X_scaled = min_max_scale(X_raw).T  # (N, dim), sklearn 관례

    gpr_full = make_gpr(seed=0)
    gpr_full.fit(X_scaled, Y)
    train_r2 = r2_score(Y, gpr_full.predict(X_scaled))
    print(f"전체 표본(90) 학습 R2(참고용, 과신 금지): {train_r2:.4f}")

    r2s = kfold_cv_r2(X_scaled, Y, k=5, seed=0)
    print(f"\n5-fold CV R2 (seed=0): {[round(v, 3) for v in r2s]}")
    print(f"평균 {r2s.mean():.3f}, 표준편차 {r2s.std():.3f}")

    cv_means = [kfold_cv_r2(X_scaled, Y, k=5, seed=s).mean() for s in range(10)]
    print(f"\n10개 시드 평균 CV R2: {[round(v, 3) for v in cv_means]}")
    print(f"평균 {np.mean(cv_means):.3f}, 표준편차 {np.std(cv_means):.3f}")
    print(
        "\n비교: PDD 10-seed 검증 R2 -0.50~+0.36 (불안정), "
        "선형회귀 10-seed 평균 CV R2 -0.065±0.073 (일관되게 낮음)"
    )


if __name__ == "__main__":
    main()
