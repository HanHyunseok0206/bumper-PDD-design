"""
pdd_interaction_ablation_20261007.py

목적: 팀원이 돌린 PDD 결과(상호작용 y=2, 전체 데이터 Adj R^2=0.63)가 과적합이라고 판단한 근거를
      직접 재현/검증. 4차 DOE(N=888, 노이즈 U1~U3 포함)로 두 가지를 비교:

        1) 전체 22변수에서 y=1(상호작용 없음) vs y=2(2변수 상호작용 포함) 홀드아웃 R^2
        2) 파라미터 수가 문제인지 확인하기 위해, PDD 민감도 분석 상위 변수만 남긴 축소 11변수
           (B4,B9,C2,C4,B1,B3,B6,C7 설계변수 8개 + U1,U2,U3 노이즈 3개)로 다시 1)과 동일 비교

      find_optimal_degree에 patience를 크게 줘서 조기 종료 없이 (n, y) 조합 전부를 로그로
      받아온 뒤, 거기서 y=1/y=2를 나란히 비교.

      결론: 파라미터를 줄여도(11변수) 상호작용을 켜는 순간 차수와 무관하게 항상 검증 R^2가
      더 나빠짐 -> "표본 부족" 문제가 아니라 "지금 데이터 규모로는 상호작용 신호 자체가
      노이즈보다 약하다"는 뜻으로 해석. 상세 해석은 docs/GUIDELINE.md, docs/PAPER.md 3.4절 참고.
"""

import pandas as pd

from pdd_first_pass_20260922 import min_max_scale, find_optimal_degree
from pdd_fourth_pass_20261001 import load_clean

FULL_COLS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9",
             "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "CA1", "CA2",
             "U1", "U2", "U3"]
REDUCED_COLS = ["B4", "B9", "C2", "C4", "B1", "B3", "B6", "C7", "U1", "U2", "U3"]


def sweep(X_scaled, Y, max_n, label):
    _, _, _, log = find_optimal_degree(X_scaled, Y, max_n=max_n, max_y=2, patience=999, seed=0)
    by_n = {}
    for n, y, k, r2 in log:
        by_n.setdefault(n, {})[y] = (k, r2)

    print(f"\n{label} (dim={X_scaled.shape[0]}):")
    print(f"  {'n':>2} | {'y=1 (상호작용 없음)':>20} | {'y=2 (상호작용 포함)':>20}")
    for n in sorted(by_n):
        y1 = by_n[n].get(1)
        y2 = by_n[n].get(2)
        r1 = f"{y1[1]:.3f}" if y1 else "-"
        r2v = f"{y2[1]:.3f}" if y2 else "-"
        print(f"  {n:>2} | {r1:>20} | {r2v:>20}")


def main():
    df = load_clean("doe_results_raw_20261001.csv")
    Y = df["SEA"].to_numpy()

    X_full = min_max_scale(df[FULL_COLS].to_numpy().T)
    sweep(X_full, Y, max_n=2, label="전체 22변수")

    X_reduced = min_max_scale(df[REDUCED_COLS].to_numpy().T)
    sweep(X_reduced, Y, max_n=4, label="축소 11변수 (B4,B9,C2,C4,B1,B3,B6,C7,U1,U2,U3)")


if __name__ == "__main__":
    main()
