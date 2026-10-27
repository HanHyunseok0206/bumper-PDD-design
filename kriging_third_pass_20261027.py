"""
kriging_third_pass_20261027.py

목적: 7차 DOE(doe_results_raw_20261022.csv, N=1482, 수정 SEA 기준 상위 변수 집중재해석)로
      크리깅 서로게이트를 학습하고 홀드아웃/K-fold 교차검증 R^2로 품질 확인.

      1차(kriging_first_pass, N=90)·2차(kriging_second_pass, N=603)는 19변수 전체를 그대로
      써서 등방성(isotropic) Matern 커널로도 CV R^2가 0.17~0.18에 그쳤고, "변수별 length_scale을
      따로 학습하는 비등방성(ARD) 커널로 교체 시도"가 다음 단계로 보류돼 있었음(2026-09-23 메모).
      이번엔 변수가 12개로 줄고 PDD 스크리닝까지 거친 데이터라 ARD 커널을 바로 시도하고,
      등방성 커널과 CV R^2로 비교해서 어느 쪽을 최종 서로게이트로 쓸지 판단함.

      스케일링: GUIDELINE.md 지침대로 "실제 설계변수 범위(LB/UB)를 알고 있으면 표본 min/max가
      아니라 이론적 정의역으로 스케일링"해야 하나, 이번 DOE는 팀원이 처음부터 의도적으로 설정한
      범위를 LHS로 채운 것이라(각 변수가 둥근 비율로 시작/끝 — 예: B2 2.9001~3.0999) 표본 min/max가
      곧 이론적 정의역과 거의 같다고 보고 그대로 사용. 차이가 있다면 이번 DOE가 실제로 쓰려는
      설계공간 자체를 좁게/넓게 재정의한 결과이지, 이론값을 못 채운 샘플링 한계가 아님.
"""

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

VAR_COLS = ["B2", "B3", "B5", "B6", "B8", "B9", "C3", "C5", "C7", "U1", "U2", "U3"]
NOISE_NAMES = {"U1", "U2", "U3"}
DESIGN_IDX = [i for i, c in enumerate(VAR_COLS) if c not in NOISE_NAMES]
NOISE_IDX = [i for i, c in enumerate(VAR_COLS) if c in NOISE_NAMES]


def theoretical_scale(x, domain_min, domain_max, target_min=-1, target_max=1):
    x_std = (x - domain_min) / (domain_max - domain_min)
    return x_std * (target_max - target_min) + target_min


def load_clean(csv_path):
    df = pd.read_csv(csv_path)
    assert (df["SEA"] == 0).sum() == 0
    assert df.isna().sum().sum() == 0
    assert df["Trial"].duplicated().sum() == 0
    return df


def make_gpr(kernel_kind, dim, seed=0, n_restarts=2):
    if kernel_kind == "isotropic":
        length_scale = 1.0
    elif kernel_kind == "ard":
        length_scale = np.ones(dim)
    else:
        raise ValueError(kernel_kind)
    kernel = ConstantKernel(1.0) * Matern(length_scale=length_scale, nu=2.5) + WhiteKernel(1e-3)
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=n_restarts, random_state=seed)


def r2_score(Y, pred):
    ss_res = np.sum((Y - pred) ** 2)
    ss_tot = np.sum((Y - Y.mean()) ** 2)
    return 1 - ss_res / ss_tot


def kfold_cv_r2(X, Y, kernel_kind, k=5, seed=0, verbose=False):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    r2s = []
    for i in range(k):
        val_idx = folds[i]
        tr_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        gpr = make_gpr(kernel_kind, X.shape[1], seed=seed)
        gpr.fit(X[tr_idx], Y[tr_idx])
        pred = gpr.predict(X[val_idx])
        r2s.append(r2_score(Y[val_idx], pred))
        if verbose:
            print(f"    [{kernel_kind} seed={seed} fold={i}] R2={r2s[-1]:.3f}", flush=True)
    return np.array(r2s)


def main():
    df = load_clean("doe_results_raw_20261022.csv")
    X_raw = df[VAR_COLS].to_numpy()
    Y = df["SEA"].to_numpy()

    domain_min = X_raw.min(axis=0)
    domain_max = X_raw.max(axis=0)
    X_scaled = theoretical_scale(X_raw, domain_min, domain_max)

    print(f"N={len(df)}, dim={len(VAR_COLS)} (설계 {len(DESIGN_IDX)} + 불확실성 {len(NOISE_IDX)})")
    print(f"변수 순서: {VAR_COLS}")

    n_seeds = 5
    for kernel_kind in ["isotropic", "ard"]:
        print(f"\n{'='*60}\n커널: {kernel_kind}\n{'='*60}", flush=True)
        cv_means = [float(kfold_cv_r2(X_scaled, Y, kernel_kind, k=5, seed=s, verbose=True).mean()) for s in range(n_seeds)]
        print(f"{n_seeds}개 시드 평균 CV R2: {[round(v, 3) for v in cv_means]}")
        print(f"  평균 {np.mean(cv_means):.4f}, 표준편차 {np.std(cv_means):.4f}")

    print(f"\n{'='*60}\n최종 모델(ARD, 전체 데이터로 학습)\n{'='*60}")
    gpr_final = make_gpr("ard", X_scaled.shape[1], seed=0, n_restarts=5)
    gpr_final.fit(X_scaled, Y)
    train_r2 = r2_score(Y, gpr_final.predict(X_scaled))
    print(f"학습 R2: {train_r2:.4f} (참고용 — 과적합 가능, CV R2로 판단)")
    print(f"학습된 커널: {gpr_final.kernel_}")
    print(f"domain_min: {dict(zip(VAR_COLS, domain_min.round(4)))}")
    print(f"domain_max: {dict(zip(VAR_COLS, domain_max.round(4)))}")


if __name__ == "__main__":
    main()
