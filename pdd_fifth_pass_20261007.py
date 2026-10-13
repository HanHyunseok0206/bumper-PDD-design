"""
pdd_fifth_pass_20261007.py

목적: 팀원이 전달한 5차 DOE 결과(doe_results_raw_20261007.csv, simulation_results_lhs_1007.csv
      원본, 2932 trial)로 PDD 민감도 분석. 4차(888개)와 변수 구성이 동일(22개: 설계변수 19 +
      노이즈 U1~U3)해서 pdd_fourth_pass_20261001.py의 VAR_COLS/run_pdd_generic을 그대로 재사용.

      4차 대비 특징:
        - 표본 2932개 (4차의 약 3.3배)
        - 변수 범위는 4차와 거의 동일(B1~C8 ~6.66%, CA1 1.25%, CA2 0.95%, U1~U3 ~10%) — 새로
          범위를 바꾼 게 아니라 같은 설계로 표본만 더 뽑은 것으로 보임(파일명의 "lhs"가
          Latin Hypercube Sampling을 가리키는 것으로 추정)
        - Trial 번호가 2962까지 가는데 2932개만 있어 30개 결측(메싱/해석 실패로 추정, 3차 때도
          비슷한 패턴 있었음)
        - SEA=0/Internal_Energy 이상치 없음, 전량 유효
"""

import numpy as np
import pandas as pd

from pdd_fourth_pass_20261001 import VAR_COLS, NOISE_NAMES, load_clean, run_pdd_generic


def main():
    df4 = load_clean("doe_results_raw_20261001.csv")
    df5 = load_clean("doe_results_raw_20261007.csv")

    res4 = run_pdd_generic(df4, VAR_COLS, "4차 DOE (2026-10-01, N=888)")
    res5 = run_pdd_generic(df5, VAR_COLS, "5차 DOE (2026-10-07, N=2932)")

    print(f"\n{'='*70}\n4차 vs 5차 비교 요약\n{'='*70}")
    for label, res in [("4차", res4), ("5차", res5)]:
        print(f"{label}: N={res['N']:5d}, k={res['k']:3d}(k/N={res['k']/res['N']:.4f}), "
              f"학습R2={res['train_r2']:.3f}, CV R2평균={res['cv_mean']:.3f}"
              f"±{np.std(res['r2s']):.3f}, gap={res['gap']:.3f} (n={res['best_n']},y={res['best_y']})")

    top4 = {name for name, _ in res4["rows"][:6]}
    top5 = {name for name, _ in res5["rows"][:6]}
    print(f"\n4차 top6: {top4}")
    print(f"5차 top6: {top5}")
    print(f"교집합: {top4 & top5}")

    noise4 = {n: v for n, v in res4["rows"] if n in NOISE_NAMES}
    noise5 = {n: v for n, v in res5["rows"] if n in NOISE_NAMES}
    print(f"\n노이즈 변수 기여율 4차: {noise4}")
    print(f"노이즈 변수 기여율 5차: {noise5}")


if __name__ == "__main__":
    main()
