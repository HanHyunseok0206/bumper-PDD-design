"""
pdd_sixth_pass_20261014.py

목적: 팀원이 2026-10-14에 전달한 집중 재해석 DOE(simulation_results_2nd_1014.csv, 원본을
      doe_results_raw_20261014.csv로 보관, 778 trial)를 독립적으로 재검증.

      이 DOE는 지금까지(Case 1~5)와 성격이 다름 — 설계변수 19개 전체가 아니라, PDD 스크리닝에서
      상위로 꼽힌 B1, B4, B6, B9, C2, C4 6개 + 불확실성 변수 U1~U3 3개, 총 9변수만 흔들고
      나머지 13개 설계변수는 (아마 공칭값으로) 고정한 상태로 돌린 집중 재해석. 범위 폭은
      기존과 동일(B*~6.6%, U1~U3~10%) — 범위를 넓힌 게 아니라 변수 수만 줄인 것.

      팀원이 보낸 PDF(1014 미팅 자료)에는 n=3,y=2(상호작용 포함)로 돌린 자체 분석 결과가
      있는데, Train R²=0.718 / Holdout R²=0.598이고 B6가 55.4%로 압도적 1위로 나옴.
      B6는 Case 3~5(19변수 풀 스크리닝)에서는 top5 바깥이거나(Case 3 3.7%, Case 4 2.9%) Case 5에서
      겨우 4위(4.7%) 수준이었어서, 이 결과가 진짜인지 아니면 10/12 미팅 때처럼 상호작용을
      켜서 생긴 과적합인지 확인이 필요해 재실행함.

      결과: find_optimal_degree가 상호작용 유무와 무관하게 S=1(상호작용 없음)을 선택 —
      즉 상호작용을 끈 모델이 이미 최선이라 teammate의 y=2 결과도 사실상 상호작용 항의
      기여가 미미했을 것으로 추정. m=4, k/N=0.048로 매우 안정적이고, 10-seed CV R²=
      0.591±0.030, gap=0.037(train 0.628 - cv 0.591)로 과적합 신호 없음. **B6 58.3% 1위
      독립 재현됨 — 10/12 때와 달리 이번엔 과적합이 아니라 진짜 신호로 판단.**

      다만 19변수 풀 스크리닝(Case 3~5) 대비 B6의 비중이 비정상적으로 크게 뛴 점은 단순 재정규화
      (13개 변수를 고정해서 남은 변수들 분모가 줄어든 효과)만으로는 설명이 안 됨 — Case 5 기준
      B6(4.7%)를 B1+B4+B6+B9+C2+C4+U1+U2+U3의 합(~84%)으로 재정규화해도 5.6%에 그쳐
      58%와는 10배 이상 차이. B6가 다른 (지금 고정된) 설계변수와 상호작용해야만 드러나는
      효과이거나, 전역 저차 PDD(m=5, 19변수 동시 적합)가 B6의 비선형 주효과 자체를 놓치고
      있었을 가능성 — **미확정, 후속 질문 필요**(아래 GUIDELINE.md 체크리스트 참고).
"""

import numpy as np
import pandas as pd

from pdd_first_pass_20260922 import PDD, min_max_scale, find_optimal_degree, get_sobol2_fixed

VAR_COLS = ["B1", "B4", "B6", "B9", "C2", "C4", "U1", "U2", "U3"]
NOISE_NAMES = {"U1", "U2", "U3"}


def load_clean(csv_path):
    df = pd.read_csv(csv_path)
    assert (df["SEA"] == 0).sum() == 0
    return df


def run_pdd(df, var_cols, label, max_n=6, max_y=2, n_seeds=10):
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

    print(f"\n{'='*60}\n{label} (N={len(df)}, m_max={max_n}, S_max={max_y})\n{'='*60}")
    print(f"선택: m={best_n}, S={best_y}  (파라미터 {k}개, k/N={k/N:.3f})")
    print(f"{n_seeds}개 시드 검증 R2: {[round(v, 3) for v in r2s]}")
    print(f"  평균 {cv_mean:.3f}, 표준편차 {np.std(r2s):.3f}")
    print(f"전체 학습 R2: {train_r2:.4f}  (gap={gap:.3f})")
    print("민감도 지수(SEA 분산 기여율):")
    for names, val in rows:
        flag = " *noise" if any(n in NOISE_NAMES for n in names.split("+")) else ""
        print(f"  {names:12s} {val:.4f}{flag}")

    return {"rows": rows, "best_n": best_n, "best_y": best_y, "train_r2": train_r2,
            "cv_mean": cv_mean, "gap": gap, "k": k, "N": N, "r2s": r2s}


def main():
    df = load_clean("doe_results_raw_20261014.csv")

    # S=1 고정(기존 Case 1~5와 같은 조건)과 teammate가 쓴 S<=2 둘 다 확인 —
    # find_optimal_degree가 둘 중 뭘 선택하는지가 "상호작용이 진짜 필요한가"의 답.
    res_s1 = run_pdd(df, VAR_COLS, "Case 6 집중 재해석 (2026-10-14, S=1 고정)", max_y=1)
    res_s2 = run_pdd(df, VAR_COLS, "Case 6 집중 재해석 (2026-10-14, S<=2 허용, teammate 조건)", max_y=2)

    print(f"\n{'='*60}\n결론\n{'='*60}")
    print(f"S=1 고정: m={res_s1['best_n']}, CV R2={res_s1['cv_mean']:.3f}±{np.std(res_s1['r2s']):.3f}")
    print(f"S<=2 허용: 선택된 S={res_s2['best_y']} (S=2가 선택 안 됐으면 상호작용 불필요 확인)")
    print("B6 1위 재현 여부:", "재현됨" if res_s1["rows"][0][0] == "B6" else "재현 안 됨")


if __name__ == "__main__":
    main()
