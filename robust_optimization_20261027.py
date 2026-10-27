"""
robust_optimization_20261027.py

목적: kriging_third_pass_20261027.py에서 검증한 ARD 커널 크리깅(7차 DOE,
      doe_results_raw_20261022.csv, N=1482, CV R2=0.368±0.009)을 최종 서로게이트로 써서
      강건 목적함수 J(x) = mean(SEA) - kappa*std(SEA)를 구성하고, kappa를 0~3으로 스윕하며
      DESIGN_IDX(B2,B3,B5,B6,B8,B9,C3,C5,C7)에 대해 최적화.

      GUIDELINE.md 9~10단계. 질량 제약은 이번엔 넣지 않음(상한값이 아직 합의되지 않아서
      2026-10-27 대화에서 보류하기로 함) — 대신 최적점의 예측 질량을 참고용으로만 같이 출력.
      불확실성 변수(NOISE_IDX=U1,U2,U3)는 DOE가 각 변수 구간에서 LHS로 균등하게 뽑힌 것으로
      보여서 몬테카를로 표본도 스케일링된 [-1,1] 구간에서 균등분포로 추출.

      주의: 여기서 나오는 최적점은 서로게이트 예측일 뿐, 실제 카티아/아바쿠스 재해석으로
      검증 전까지는 확정 결과가 아님(11단계). 팀원에게 넘길 때 그렇게 명시할 것.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from kriging_third_pass_20261027 import (
    VAR_COLS, NOISE_NAMES, DESIGN_IDX, NOISE_IDX,
    theoretical_scale, load_clean, make_gpr,
)

KAPPAS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
N_MC = 3000
N_RESTARTS_OPT = 8


def inverse_scale(x_scaled, domain_min, domain_max, target_min=-1, target_max=1):
    x_std = (x_scaled - target_min) / (target_max - target_min)
    return x_std * (domain_max - domain_min) + domain_min


def estimate_robust_sea(d, model, n_mc=N_MC, seed=0):
    dim = len(DESIGN_IDX) + len(NOISE_IDX)
    rng = np.random.default_rng(seed)
    x_full = np.zeros((n_mc, dim))
    x_full[:, DESIGN_IDX] = np.array(d).reshape(1, -1)
    x_full[:, NOISE_IDX] = rng.uniform(-1, 1, size=(n_mc, len(NOISE_IDX)))
    sea_samples = model.predict(x_full)
    return sea_samples.mean(), sea_samples.std()


def robust_objective(d, kappa, model):
    mean_sea, std_sea = estimate_robust_sea(d, model)
    return -(mean_sea - kappa * std_sea)


def multi_start_optimize(kappa, model, rng, n_restarts=N_RESTARTS_OPT):
    bounds = [(-1, 1)] * len(DESIGN_IDX)
    best = None
    for i in range(n_restarts):
        d0 = rng.uniform(-1, 1, size=len(DESIGN_IDX)) if i > 0 else np.zeros(len(DESIGN_IDX))
        res = minimize(robust_objective, d0, args=(kappa, model), method="SLSQP", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    return best


def main():
    df = load_clean("doe_results_raw_20261022.csv")
    X_raw = df[VAR_COLS].to_numpy()
    Y = df["SEA"].to_numpy()
    Mass = df["Total_Mass"].to_numpy()

    domain_min = X_raw.min(axis=0)
    domain_max = X_raw.max(axis=0)
    X_scaled = theoretical_scale(X_raw, domain_min, domain_max)

    print("최종 ARD 크리깅 학습 중...")
    gpr = make_gpr("ard", X_scaled.shape[1], seed=0, n_restarts=5)
    gpr.fit(X_scaled, Y)
    print(f"학습 완료. 커널: {gpr.kernel_}")

    # 참고용 질량 예측: 선형회귀로 충분(질량은 두께류 변수에 대체로 선형적)
    A = np.hstack([X_scaled, np.ones((X_scaled.shape[0], 1))])
    mass_coef, *_ = np.linalg.lstsq(A, Mass, rcond=None)

    def predict_mass(d_design):
        x_full = np.zeros(len(VAR_COLS))
        x_full[DESIGN_IDX] = d_design
        return float(np.dot(np.append(x_full, 1.0), mass_coef))

    # 기준점(모든 변수 중앙값, d=0)
    base_mean, base_std = estimate_robust_sea(np.zeros(len(DESIGN_IDX)), gpr, seed=0)
    print(f"\n기준 설계(전체 중앙값, d=0): mean(SEA)={base_mean:.1f}, std(SEA)={base_std:.1f}, "
          f"예측질량={predict_mass(np.zeros(len(DESIGN_IDX))):.4f}")

    rng = np.random.default_rng(0)
    print(f"\n{'='*70}\nkappa 스윕 결과\n{'='*70}")
    rows = []
    for kappa in KAPPAS:
        res = multi_start_optimize(kappa, gpr, rng)
        d_opt = res.x
        mean_sea, std_sea = estimate_robust_sea(d_opt, gpr, seed=1)  # 평가용 다른 seed로 재확인
        J = mean_sea - kappa * std_sea
        x_opt_physical = inverse_scale(d_opt, domain_min[DESIGN_IDX], domain_max[DESIGN_IDX])
        mass_opt = predict_mass(d_opt)
        rows.append({
            "kappa": kappa, "mean_SEA": mean_sea, "std_SEA": std_sea, "J": J,
            "mass_pred": mass_opt, "d_scaled": d_opt, "x_physical": x_opt_physical,
        })
        print(f"\nkappa={kappa}: mean={mean_sea:.1f}, std={std_sea:.1f}, J={J:.1f}, "
              f"예측질량={mass_opt:.4f}")
        design_names = [VAR_COLS[i] for i in DESIGN_IDX]
        for name, val in zip(design_names, x_opt_physical):
            print(f"    {name:4s} = {val:.4f}")

    print(f"\n{'='*70}\n요약표\n{'='*70}")
    print(f"{'kappa':>6} {'mean(SEA)':>12} {'std(SEA)':>10} {'J':>10} {'mass':>8}")
    for r in rows:
        print(f"{r['kappa']:>6.1f} {r['mean_SEA']:>12.1f} {r['std_SEA']:>10.1f} {r['J']:>10.1f} {r['mass_pred']:>8.4f}")

    return rows, domain_min, domain_max


if __name__ == "__main__":
    main()
