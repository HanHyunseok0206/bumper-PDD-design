"""
pdd_seventh_pass_20261027.py

목적: 팀원이 "수정된 SEA 산출 방식"(Pure_Absorbed_Energy 기반, ALLAE 제외) 기준
      민감도 상위 변수로 추려서 집중 재해석한 7차 DOE(doe_results_raw_20261022.csv,
      원본 simulation_results_2nd_1022.csv, 1482 trial)로 PDD 민감도 분석.

      6차(doe_results_raw_20261014.csv, Internal_Energy 기반 옛 SEA)와 달리, 이번엔
      설계변수 19개가 아니라 B2,B3,B5,B6,B8,B9,C3,C5,C7 9개만 흔들고 나머지 10개는 고정 +
      불확실성 변수 U1~U3은 항상 포함(12변수). SEA 평균이 ~70792로 "수정 SEA" 스케일과
      일치해서 이번 재해석부터 교정된 산출 방식이 쓰였음을 확인함(대화 기록 참고).

      변수 집합이 6차(B1,B4,B6,B9,C2,C4+U1~U3)와도 다름 — 6차는 "옛 SEA" 기준 상위 변수였고
      이번은 "수정 SEA"로 Case5 동일설계를 재분석했을 때 나온 top6(B9,B2,U1,C7,B8,B5)를 그대로
      포함하고 B3,C3,C5를 여유분으로 추가한 조합. 두 SEA 정의가 서로 다른 민감도 구조를 보여서
      (교체 당시 B4/C2/C4 탈락, U1 급상승 확인됨) 변수 선택 기준이 달라진 것은 당연함.
"""

import numpy as np
import pandas as pd

from pdd_first_pass_20260922 import PDD, min_max_scale, find_optimal_degree, get_sobol2_fixed

VAR_COLS = ["B2", "B3", "B5", "B6", "B8", "B9", "C3", "C5", "C7", "U1", "U2", "U3"]
NOISE_NAMES = {"U1", "U2", "U3"}


def load_clean(csv_path):
    df = pd.read_csv(csv_path)
    assert (df["SEA"] == 0).sum() == 0, "SEA=0 존재"
    assert df.isna().sum().sum() == 0, "NaN 존재"
    assert df["Trial"].duplicated().sum() == 0, "중복 Trial 존재"
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
    print(f"seed=0 선택: m={best_n}, S={best_y}, 검증 R2={best_r2:.4f}  (파라미터 {k}개, k/N={k/N:.3f})")
    print(f"{n_seeds}개 시드 검증 R2: {[round(v, 3) for v in r2s]}")
    print(f"  평균 {cv_mean:.3f}, 표준편차 {np.std(r2s):.3f}")
    print(f"전체 학습 R2: {train_r2:.4f}  (gap={gap:.3f})")
    print(f"\n민감도 지수(SEA 분산 기여율) 전체 {len(rows)}개 (불확실성 변수는 * 표시):")
    for names, val in rows:
        flag = " *" if any(n in NOISE_NAMES for n in names.split("+")) else ""
        print(f"  {names:12s} {val:.4f}{flag}")

    return {"rows": rows, "best_n": best_n, "best_y": best_y, "train_r2": train_r2,
            "cv_mean": cv_mean, "gap": gap, "k": k, "N": N, "r2s": r2s}


def main():
    df = load_clean("doe_results_raw_20261022.csv")
    res = run_pdd_generic(df, VAR_COLS, "7차 DOE (2026-10-22, 수정 SEA 기준 상위 변수 집중재해석, N=1482)")

    noise_rows = [(n, v) for n, v in res["rows"] if n in NOISE_NAMES]
    design_rows = [(n, v) for n, v in res["rows"] if n not in NOISE_NAMES]
    noise_total = sum(v for _, v in noise_rows)
    print(f"\n불확실성 변수(U1~U3) 합산 기여율: {noise_total:.4f} ({noise_total*100:.1f}%)")
    print("불확실성 변수 개별:", {n: round(v, 4) for n, v in noise_rows})
    print(f"\n설계변수 상위 (누적 기여율 확인용):")
    cum = 0
    for n, v in design_rows:
        cum += v
        print(f"  {n:6s} {v:.4f}  (누적 {cum:.4f})")


if __name__ == "__main__":
    main()
