"""
kriging_second_pass_20260923.py

목적: 2차 DOE 결과(doe_results_raw_20260923.csv, 603 trial, 설계변수 범위 확대판)로 크리깅을
      재검증하고 1차(kriging_first_pass_20260922.py, N=90) 결과와 비교.

      주의: 이 스크립트는 아직 PDD 민감도 스크리닝 결과를 반영하지 않고 19변수 전체를 그대로 씀.
      (원래 계획은 "스크리닝 후 변수를 줄여 같은 데이터로 재학습"이지, 새 FE를 도는 게 아님 —
      GUIDELINE.md 7단계 참고. 여기서는 우선 "스크리닝 적용 전" 베이스라인만 잡아둠.)
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from pdd_first_pass_20260922 import VAR_COLS, min_max_scale


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


def run_kriging(df, label, n_seeds=10):
    X_raw = df[VAR_COLS].to_numpy().T
    Y = df["SEA"].to_numpy()
    X_scaled = min_max_scale(X_raw).T

    gpr_full = make_gpr(seed=0)
    gpr_full.fit(X_scaled, Y)
    train_r2 = float(r2_score(Y, gpr_full.predict(X_scaled)))

    r2s_seed0 = kfold_cv_r2(X_scaled, Y, k=5, seed=0)
    cv_means = [float(kfold_cv_r2(X_scaled, Y, k=5, seed=s).mean()) for s in range(n_seeds)]

    print(f"\n{'='*60}\n{label} (N={len(df)})\n{'='*60}")
    print(f"전체 학습 R2: {train_r2:.4f}  (WhiteKernel이 noise_level 하한에 붙으면 "
          f"보간에 가까워져 1에 근접할 수 있음 — 참고용, CV R2로 판단할 것)")
    print(f"5-fold CV R2 (seed=0): {[round(float(v), 3) for v in r2s_seed0]}")
    print(f"{n_seeds}개 시드 평균 CV R2: {[round(v, 3) for v in cv_means]}")
    print(f"  평균 {np.mean(cv_means):.3f}, 표준편차 {np.std(cv_means):.3f}")

    return {"train_r2": train_r2, "cv_means": cv_means}


def main():
    import pandas as pd

    df1 = pd.read_csv("doe_results_raw_20260922.csv")
    df1 = df1[df1["SEA"] != 0].copy()
    median_ie1 = df1["Internal_Energy"].median()
    df1 = df1[df1["Internal_Energy"] > median_ie1 * 0.5].reset_index(drop=True)

    df2 = pd.read_csv("doe_results_raw_20260923.csv")

    res1 = run_kriging(df1, "1차 DOE (N=90)")
    res2 = run_kriging(df2, "2차 DOE (N=603)")

    print(f"\n{'='*60}\n비교 요약\n{'='*60}")
    print(f"1차: 10-seed 평균 CV R2 = {np.mean(res1['cv_means']):.3f} ± {np.std(res1['cv_means']):.3f}")
    print(f"2차: 10-seed 평균 CV R2 = {np.mean(res2['cv_means']):.3f} ± {np.std(res2['cv_means']):.3f}")


if __name__ == "__main__":
    main()
