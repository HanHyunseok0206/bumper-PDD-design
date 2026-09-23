"""
pdd_second_pass_20260923.py

목적: 팀원이 전달한 2차 DOE 결과(doe_results_raw_20260923.csv, 603 trial)로 PDD 민감도 분석을
      재실행하고, 1차(90개, doe_results_raw_20260922.csv)와 비교. 함수(basis/PDD/get_sobol2_fixed/
      find_optimal_degree/min_max_scale)는 pdd_first_pass_20260922.py와 동일, import해서 재사용.

      2차 데이터 특징(1차 대비):
        - 표본 603개 (1차는 클리닝 후 90개, 약 6~7배)
        - 설계변수 변동폭이 B1~C8은 공칭값 대비 ~13%(1차 ~1.7~3.3%)로 대폭 확대됨.
          단 C9(~1.24%)·C10(~0.95%)은 오히려 1차보다도 더 좁음 — 의도한 건지 확인 필요.
        - SEA=0/Internal_Energy 이상치 없음 (1차는 9+1건 제외 후 90개였음)
        
"""

import numpy as np
from pdd_first_pass_20260922 import (
    VAR_COLS, PDD, basis, find_optimal_degree, get_sobol2_fixed, min_max_scale,
)


def load_second_pass(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    assert (df["SEA"] == 0).sum() == 0
    median_ie = df["Internal_Energy"].median()
    outliers = df[df["Internal_Energy"] <= median_ie * 0.5]
    assert len(outliers) == 0, f"IE 이상치 {len(outliers)}건 발견 - 확인 필요"
    return df


def run_pdd(df, label, max_n=6, max_y=2, n_seeds=10):
    X_raw = df[VAR_COLS].to_numpy().T
    Y = df["SEA"].to_numpy()
    X_scaled = min_max_scale(X_raw)
    N = X_scaled.shape[1]

    best_n, best_y, best_r2, log = find_optimal_degree(X_scaled, Y, max_n=max_n, max_y=max_y, seed=0)
    # np.float64를 그대로 리스트에 담으면 repr이 "np.float64(0.123)"으로 찍혀서 지저분함 -> float()로 캐스팅
    r2s = [float(find_optimal_degree(X_scaled, Y, max_n=max_n, max_y=max_y, seed=s)[2]) for s in range(n_seeds)]

    exp_full, mapping = PDD(X_scaled, best_n, best_y)
    k = exp_full.shape[1]  # 절편 포함 파라미터(회귀계수) 개수
    Ci = np.linalg.pinv(exp_full) @ Y
    train_r2 = float(1 - np.sum((Y - exp_full @ Ci) ** 2) / np.sum((Y - Y.mean()) ** 2))
    cv_mean = float(np.mean(r2s))
    gap = train_r2 - cv_mean

    sens = get_sobol2_fixed(Ci, mapping, exp_full)
    rows = []
    for key, val in sens.items():
        idxs = [int(s) for s in key[1:].split("_")]
        names = "+".join(VAR_COLS[i - 1] for i in idxs)
        rows.append((names, val))
    rows.sort(key=lambda r: -r[1])

    print(f"\n{'='*60}\n{label} (N={len(df)})\n{'='*60}")
    print(f"seed=0 선택: n={best_n}, y={best_y}, 검증 R2={best_r2:.4f}  (파라미터 {k}개, k/N={k/N:.3f})")
    # 10개 시드 각각 다른 방식으로 학습/검증 데이터를 나눠서(find_optimal_degree 내부의 랜덤 분할 seed)
    # 표본이 적을 때 분할 하나로 운 좋게/나쁘게 나오는 걸 방지하려고 반복 검증한 값들.
    print(f"{n_seeds}개 시드 검증 R2 (분할을 seed=0~{n_seeds-1}로 바꿔가며 반복): "
          f"{[round(v, 3) for v in r2s]}")
    print(f"  평균 {cv_mean:.3f}, 표준편차 {np.std(r2s):.3f}")
    print(f"전체 학습 R2: {train_r2:.4f}")
    print(f"  학습R2 - CV평균R2 = {gap:.3f}  (이 차이가 클수록 노이즈를 외운 과적합 의심 "
          f"— 학습 R2 자체가 아니라 이 gap과 CV R2로 신뢰도를 판단할 것)")
    print(f"\n민감도 지수(SEA 분산 기여율) 전체 {len(rows)}개:")
    for names, val in rows:
        print(f"  {names:12s} {val:.4f}")

    return {"r2s": r2s, "rows": rows, "best_n": best_n, "best_y": best_y,
            "train_r2": train_r2, "cv_mean": cv_mean, "gap": gap, "k": k, "N": N}


def main():
    import pandas as pd

    df1 = pd.read_csv("doe_results_raw_20260922.csv")
    df1 = df1[df1["SEA"] != 0].copy()
    median_ie1 = df1["Internal_Energy"].median()
    df1 = df1[df1["Internal_Energy"] > median_ie1 * 0.5].reset_index(drop=True)

    df2 = load_second_pass("doe_results_raw_20260923.csv")

    res1 = run_pdd(df1, "1차 DOE (2026-09-22, ISO 공차 범위)")
    res2 = run_pdd(df2, "2차 DOE (2026-09-23, 확대 범위)")

    print(f"\n{'='*60}\n비교 요약\n{'='*60}")
    print(f"1차: N={res1['N']}, 파라미터 {res1['k']}개(k/N={res1['k']/res1['N']:.3f}), "
          f"학습 R2={res1['train_r2']:.3f}, CV R2 평균={res1['cv_mean']:.3f}±{np.std(res1['r2s']):.3f}, "
          f"gap={res1['gap']:.3f} (n={res1['best_n']}, y={res1['best_y']})")
    print(f"2차: N={res2['N']}, 파라미터 {res2['k']}개(k/N={res2['k']/res2['N']:.3f}), "
          f"학습 R2={res2['train_r2']:.3f}, CV R2 평균={res2['cv_mean']:.3f}±{np.std(res2['r2s']):.3f}, "
          f"gap={res2['gap']:.3f} (n={res2['best_n']}, y={res2['best_y']})")
    print("\n(gap = 학습R2 - CV평균R2. 1차는 k/N이 커서(파라미터 대비 표본 부족) 학습R2는 높아도 "
          "그 대부분이 과적합이었고, 2차는 k/N이 작아져 학습R2와 CV R2 차이가 줄어 더 신뢰할 수 있음)")

    top1 = {name for name, _ in res1["rows"][:5]}
    top2 = {name for name, _ in res2["rows"][:5]}
    print(f"\n1차 상위 5 민감도 변수: {top1}")
    print(f"2차 상위 5 민감도 변수: {top2}")
    print(f"겹치는 변수: {top1 & top2}")


if __name__ == "__main__":
    main()
