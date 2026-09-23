"""
design_matrix_diagnostics_20260922.py

목적: 1차 DOE 결과(doe_results_raw_20260922.csv)의 입력 설계행렬(B1~C10, 19변수) 자체를 진단.
      PDD 민감도 분석(pdd_first_pass_20260922.py)이 표본 90/변수 19 비율에서 불안정했던 원인이
      "표본 부족" 때문인지, 아니면 "변수들이 좁은 범위+상호 상관을 갖고 움직여서 실질적으로
      독립적인 방향 수가 19개보다 훨씬 적기 때문"인지 구분하기 위한 사전 점검.
      팀원에게 실제 LB/UB와 DOE 의도(전역 LHS인지 국소 섭동인지)를 확인 전이므로, 여기서는
      "무엇을 봤는지"만 정리하고 해석은 팀원 답변 이후로 미룸.

체크 항목:
  1) 변수별 공칭값(평균) 대비 변동폭(%) — SEA=0/이상치 제외 전/후 비교
  2) 입력변수 간 상관행렬 — |r| 큰 쌍이 있으면 "독립적으로 흔든 DOE"가 아닐 가능성
  3) VIF(분산팽창지수) — 다중공선성 정량화 (statsmodels 없어 수기 구현: VIF_i = 1/(1-R_i^2))
  4) PCA — 표준화된 설계행렬의 유효 차원 수 확인. 19차원인데 소수 PC로 분산 대부분 설명되면
     "19개 변수를 독립적으로 흔든 LHS"가 아니라 더 적은 자유도로 움직였다는 뜻
"""

import numpy as np
import pandas as pd

VAR_COLS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9",
            "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"]


def load_all_and_clean(csv_path):
    df_all = df = pd.read_csv(csv_path)
    df_valid = df[df["SEA"] != 0].copy()
    median_ie = df_valid["Internal_Energy"].median()
    df_valid = df_valid[df_valid["Internal_Energy"] > median_ie * 0.5].reset_index(drop=True)
    return df_all, df_valid


def range_pct_table(df, cols):
    rows = []
    for c in cols:
        v = df[c].to_numpy()
        nominal = v.mean()
        rng_pct = (v.max() - v.min()) / nominal * 100
        std_pct = v.std(ddof=1) / nominal * 100
        rows.append((c, nominal, v.min(), v.max(), rng_pct, std_pct))
    return pd.DataFrame(rows, columns=["var", "nominal(mean)", "min", "max", "range_%", "std_%"])


def vif_table(X_df):
    # VIF_i = 1 / (1 - R_i^2), R_i^2 from OLS of column i on all other columns (+ intercept)
    X = X_df.to_numpy()
    n, p = X.shape
    vifs = []
    for i in range(p):
        y = X[:, i]
        others = np.delete(X, i, axis=1)
        A = np.column_stack([np.ones(n), others])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vif = 1 / (1 - r2) if r2 < 1 else np.inf
        vifs.append(vif)
    return pd.DataFrame({"var": X_df.columns, "R2_on_others": None, "VIF": vifs}).assign(
        R2_on_others=lambda d: 1 - 1 / d["VIF"]
    )


def pca_variance_explained(X_df):
    X = X_df.to_numpy()
    Xs = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    # SVD 기반 PCA (공분산행렬 고유분해와 동일, 수치적으로 더 안정적)
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    var_explained = (S ** 2) / np.sum(S ** 2)
    cum = np.cumsum(var_explained)
    return var_explained, cum


def top_corr_pairs(corr_df, thresh=0.3):
    cols = corr_df.columns
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr_df.iloc[i, j]
            if abs(r) >= thresh:
                pairs.append((cols[i], cols[j], r))
    pairs.sort(key=lambda t: -abs(t[2]))
    return pairs


def main():
    df_all, df_valid = load_all_and_clean("doe_results_raw_20260922.csv")
    print(f"전체 {len(df_all)}개, 클리닝 후 {len(df_valid)}개로 진단 (클리닝 기준은 pdd_first_pass와 동일)")

    print("\n=== 1) 변수별 변동폭 (클리닝 후 90개 기준) ===")
    rt = range_pct_table(df_valid, VAR_COLS)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    print(rt.to_string(index=False))
    print(f"\n  range_% 요약: min={rt['range_%'].min():.2f}%, "
          f"median={rt['range_%'].median():.2f}%, max={rt['range_%'].max():.2f}%")

    X_df = df_valid[VAR_COLS]

    print("\n=== 2) 입력변수 간 상관행렬에서 |r|>=0.3인 쌍 ===")
    corr_df = X_df.corr()
    pairs = top_corr_pairs(corr_df, thresh=0.3)
    if not pairs:
        print("  없음 (|r|<0.3) — 변수들이 대체로 독립적으로 움직인 것으로 보임")
    else:
        for a, b, r in pairs:
            print(f"  {a:4s} - {b:4s}  r={r:+.3f}")

    print("\n=== 3) VIF (분산팽창지수, 통상 10 이상이면 문제, 5 이상이면 주의) ===")
    vt = vif_table(X_df).sort_values("VIF", ascending=False)
    print(vt.to_string(index=False))

    print("\n=== 4) PCA: 표준화된 설계행렬(19변수)의 유효 차원 ===")
    var_exp, cum = pca_variance_explained(X_df)
    for i, (v, c) in enumerate(zip(var_exp, cum), start=1):
        print(f"  PC{i:2d}: 분산기여 {v*100:5.1f}%, 누적 {c*100:5.1f}%")
    n90 = int(np.searchsorted(cum, 0.9) + 1)
    print(f"\n  누적분산 90%를 넘기는 데 필요한 PC 수: {n90} / {len(VAR_COLS)}")
    print("  (19에 가까우면 19개 방향으로 고르게 흔든 설계, 훨씬 적으면 실질적 자유도가 "
          "설계변수 수보다 적다는 뜻 — 국소 섭동/그룹 이동 등 의심)")


if __name__ == "__main__":
    main()
