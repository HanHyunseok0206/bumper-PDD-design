"""
pdd_fourth_pass_20261001.py

목적: 팀원이 불확실성(노이즈) 변수 U1, U2, U3을 추가해서 돌린 4차 DOE 결과
      (doe_results_raw_20261001.csv, 888 trial)로 PDD 민감도 분석.

      3차 대비 달라진 점:
        - 변수명이 C9/C10 -> CA1/CA2로 바뀜 (값 자체는 동일한 스케일, 그냥 이름 변경으로 보임)
        - 노이즈 변수 U1, U2, U3 추가 (설계변수 19개 + 노이즈 3개 = 22개). 셋 다 변동폭이
          공칭값 대비 정확히 ~10%로 동일 — 의도적으로 맞춘 것으로 보임. 물리적 의미(어떤 노이즈인지)는
          아직 팀원에게 확인 안 됨, U1~U3이라는 이름만 받음
        - SEA=0/이상치/중복/NaN 없음, 전량 유효(888개)

      pdd_second_pass_20260923.py의 run_pdd는 VAR_COLS가 19개로 하드코딩된 pdd_first_pass의
      상수를 그대로 쓰기 때문에 U1~U3을 분석에서 놓친다. 그래서 VAR_COLS를 인자로 받는
      일반화된 버전(run_pdd_generic)을 이 파일에 새로 정의해서 사용.
"""

import numpy as np
import pandas as pd

from pdd_first_pass_20260922 import PDD, min_max_scale, find_optimal_degree, get_sobol2_fixed

VAR_COLS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9",
            "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8",
            "CA1", "CA2", "U1", "U2", "U3"]
NOISE_NAMES = {"U1", "U2", "U3"}


def load_clean(csv_path):
    df = pd.read_csv(csv_path)
    assert (df["SEA"] == 0).sum() == 0
    median_ie = df["Internal_Energy"].median()
    outliers = df[df["Internal_Energy"] <= median_ie * 0.5]
    assert len(outliers) == 0, f"IE 이상치 {len(outliers)}건 발견 - 확인 필요"
    return df


def run_pdd_generic(df, var_cols, label, max_n=6, max_y=2, n_seeds=10):
    X_raw = df[var_cols].to_numpy().T
    Y = df["SEA"].to_numpy()
    X_scaled = min_max_scale(X_raw)
    N = X_scaled.shape[1]

    best_n, best_y, best_r2, log = find_optimal_degree(X_scaled, Y, max_n=max_n, max_y=max_y, seed=0)
    r2s = [float(find_optimal_degree(X_scaled, Y, max_n=max_n, max_y=max_y, seed=s)[2]) for s in range(n_seeds)]

    exp_full, mapping = PDD(X_scaled, best_n, best_y)
    k = exp_full.shape[1]
    Ci = np.linalg.pinv(exp_full) @ Y
    train_r2 = float(1 - np.sum((Y - exp_full @ Ci) ** 2) / np.sum((Y - Y.mean()) ** 2))
    cv_mean = float(np.mean(r2s))
    gap = train_r2 - cv_mean

    sens = get_sobol2_fixed(Ci, mapping, exp_full)
    rows = []
    for key, val in sens.items():
        idxs = [int(s) for s in key[1:].split("_")]
        names = "+".join(var_cols[i - 1] for i in idxs)
        rows.append((names, val))
    rows.sort(key=lambda r: -r[1])

    print(f"\n{'='*60}\n{label} (N={len(df)})\n{'='*60}")
    print(f"seed=0 선택: n={best_n}, y={best_y}, 검증 R2={best_r2:.4f}  (파라미터 {k}개, k/N={k/N:.3f})")
    print(f"{n_seeds}개 시드 검증 R2: {[round(v, 3) for v in r2s]}")
    print(f"  평균 {cv_mean:.3f}, 표준편차 {np.std(r2s):.3f}")
    print(f"전체 학습 R2: {train_r2:.4f}  (gap={gap:.3f})")
    print(f"\n민감도 지수(SEA 분산 기여율) 전체 {len(rows)}개 (노이즈 변수는 * 표시):")
    for names, val in rows:
        flag = " *" if any(n in NOISE_NAMES for n in names.split("+")) else ""
        print(f"  {names:12s} {val:.4f}{flag}")

    return {"rows": rows, "best_n": best_n, "best_y": best_y, "train_r2": train_r2,
            "cv_mean": cv_mean, "gap": gap, "k": k, "N": N, "r2s": r2s}


def main():
    df = load_clean("doe_results_raw_20261001.csv")
    res = run_pdd_generic(df, VAR_COLS, "4차 DOE (2026-10-01, 노이즈 U1~U3 포함)")

    noise_rows = [(n, v) for n, v in res["rows"] if n in NOISE_NAMES]
    design_rows = [(n, v) for n, v in res["rows"] if n not in NOISE_NAMES]
    noise_total = sum(v for _, v in noise_rows)
    print(f"\n노이즈 변수(U1~U3) 합산 기여율: {noise_total:.4f} ({noise_total*100:.1f}%)")
    print("노이즈 변수 개별:", {n: round(v, 4) for n, v in noise_rows})
    print(f"\n설계변수 상위 7개 (누적 기여율 확인용):")
    cum = 0
    for n, v in design_rows[:7]:
        cum += v
        print(f"  {n:6s} {v:.4f}  (누적 {cum:.4f})")


if __name__ == "__main__":
    main()
