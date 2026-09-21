"""
pdd_first_pass_20260922.py

목적: 팀원이 아바쿠스에서 처음 전달한 1차 DOE 결과(doe_results_raw_20260922.csv, 100 trial)로
      PDD 민감도 분석을 시도해본 첫 기록. 논문 작성 시 데이터 클리닝/노이즈 이슈 절 근거로 참고.
      docs/GUIDELINE.md "진행 상황"(2026-09-22) 항목과 함께 볼 것.

중요: 이 데이터의 변수명(B1~B9, C1~C10)은 레포의 doe_points.csv/generate_doe_geometry.py가
      쓰는 9변수 체계(t_beam, beam_radius, ...)와 다름. 각 번호가 정확히 어느 위치/파라미터인지,
      LB/UB가 얼마인지 아직 팀원 확인 전이라, 여기서는 표본 min/max로만 스케일링함
      (GUIDELINE 원칙상 이론적 정의역을 써야 하지만 아직 모름 — 확인되면 PDD_bumper_SEA.ipynb의
      theoretical_scale로 교체할 것).

핵심 함수(basis, PDD, get_sobol2, find_optimal_degree)는 PDD_bumper_SEA.ipynb /
validation_and_benchmarks/PDD_Legendre_ver4.ipynb와 동일, 수정 없음.
get_sobol2만 dim>=10에서 키가 겹치는 문제(예: "S17"이 변수17 단독인지 변수1+7 상호작용인지
구분 안 됨)가 있어 구분자를 넣은 버전(get_sobol2_fixed)으로 교체함.
"""

import numpy as np
import pandas as pd


def min_max_scale(x, min_val=-1, max_val=1):
    # 표본의 경험적 min/max로 스케일링. 실제 정의역(LB/UB)을 알면 theoretical_scale을 쓸 것.
    x_min = x.min(axis=1, keepdims=True)
    x_max = x.max(axis=1, keepdims=True)
    range_mask = (x_max - x_min) == 0
    x_max[range_mask] += 1e-8
    x_std = (x - x_min) / (x_max - x_min)
    return x_std * (max_val - min_val) + min_val


def basis(x, a):
    if a == 0:
        return np.ones_like(x)
    if a == 1:
        return x
    p_prev2 = np.ones_like(x)
    p_prev1 = x.copy()
    p_n = None
    for n in range(1, a):
        p_n = ((2 * n + 1) * x * p_prev1 - n * p_prev2) / (n + 1)
        p_prev2 = p_prev1
        p_prev1 = p_n
    return p_n


def PDD(x, n, y):
    dim, N = x.shape[0], x.shape[1]
    phi = [np.ones(N)]
    mapping_list = [[0] * dim]
    for i in range(dim):
        for j in range(1, n + 1):
            phi.append(basis(x[i, :], j))
            m = [0] * dim
            m[i] = 1
            mapping_list.append(m)
    if y >= 2:
        for i in range(2, n + 1):
            for j in range(1, i):
                for k in range(dim):
                    for l in range(k + 1, dim):
                        phi.append(basis(x[k, :], j) * basis(x[l, :], i - j))
                        m = [0] * dim
                        m[k] = 1
                        m[l] = 1
                        mapping_list.append(m)
    return np.array(phi).T, np.array(mapping_list).T


def get_sobol2_fixed(Ci, mapping, exp_input):
    # 원본 get_sobol2와 동일한 로직이되, dim>=10일 때 인덱스 문자열이 겹치지 않도록 구분자를 둠
    sd = {}
    total = 0.0
    for i in range(1, Ci.shape[0]):
        active = np.where(mapping[:, i] == 1)[0]
        if len(active) == 0:
            continue
        key = "S" + "_".join(str(a) for a in sorted(active + 1))
        c = np.var(Ci[i] * exp_input[:, i])
        sd[key] = sd.get(key, 0) + c
        total += c
    for k in sd:
        sd[k] = sd[k] / total if total > 0 else 0
    return sd


def find_optimal_degree(X, Y, max_n=6, max_y=2, val_ratio=0.2, patience=3, tol=1e-4, seed=0):
    # 학습/검증 분리 기반 차수 선택 (학습 R^2만으로 고르면 과적합됨 — CLAUDE.md 참고)
    rng = np.random.default_rng(seed)
    Nc = X.shape[1]
    perm = rng.permutation(Nc)
    n_val = max(int(Nc * val_ratio), 1)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr, Xv = X[:, tr_idx], X[:, val_idx]
    Ytr, Yv = Y[tr_idx], Y[val_idx]

    best_n, best_y, best_r2 = 1, 1, -float("inf")
    no_improve = 0
    log = []
    for n in range(1, max_n + 1):
        for y in range(1, max_y + 1):
            exp_tr, mapping = PDD(Xtr, n, y)
            k = exp_tr.shape[1]
            if k >= Xtr.shape[1] - 1:
                continue
            Ci = np.linalg.pinv(exp_tr) @ Ytr
            exp_v, _ = PDD(Xv, n, y)
            pred = exp_v @ Ci
            ss_res = np.sum((Yv - pred) ** 2)
            ss_tot = np.sum((Yv - Yv.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot
            log.append((n, y, k, r2))
            if r2 > best_r2 + tol:
                best_r2 = r2
                best_n, best_y = n, y
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                return best_n, best_y, best_r2, log
    return best_n, best_y, best_r2, log


VAR_COLS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9",
            "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"]


def load_clean_data(csv_path):
    df = pd.read_csv(csv_path)

    # 1차 필터: SEA=0 (메싱 실패, 팀원이 이미 인지한 케이스)
    df_valid = df[df["SEA"] != 0].copy()

    # 2차 필터: SEA=0은 아니지만 Internal_Energy가 나머지 대비 8배가량 낮은 Trial 1.
    # 크래시 펄스가 정상적으로 전개되지 못한 채 해석이 종료된 것으로 판단해 추가 제외함
    # (SEA=0 필터만으로는 이런 케이스가 안 걸린다는 게 이번에 확인한 점).
    median_ie = df_valid["Internal_Energy"].median()
    df_valid = df_valid[df_valid["Internal_Energy"] > median_ie * 0.5].reset_index(drop=True)

    return df_valid


def main():
    df_valid = load_clean_data("doe_results_raw_20260922.csv")
    print(f"유효 표본 수: {len(df_valid)} / 100")

    X_raw = df_valid[VAR_COLS].to_numpy().T
    Y = df_valid["SEA"].to_numpy()
    X_scaled = min_max_scale(X_raw)

    best_n, best_y, best_r2, log = find_optimal_degree(X_scaled, Y, max_n=6, max_y=2, seed=0)
    print("\n차수 선택 로그 (n, y, 계수개수, val_R2):")
    for row in log:
        print(" ", row)
    print(f"선택: n={best_n}, y={best_y}, 검증 R2={best_r2:.4f}")

    # 표본이 적어 분할에 따라 R2가 크게 흔들림 — 시드 10개로 안정성 확인
    r2s = [find_optimal_degree(X_scaled, Y, max_n=6, max_y=2, seed=s)[2] for s in range(10)]
    print(f"\n10개 시드 검증 R2: {[round(v, 3) for v in r2s]}")
    print(f"평균 {np.mean(r2s):.3f}, 표준편차 {np.std(r2s):.3f}  (불안정 — 잠정 결과로만 취급할 것)")

    exp_full, mapping = PDD(X_scaled, best_n, best_y)
    Ci = np.linalg.pinv(exp_full) @ Y
    train_r2 = 1 - np.sum((Y - exp_full @ Ci) ** 2) / np.sum((Y - Y.mean()) ** 2)
    print(f"\n전체 학습 R2(참고용, 과신 금지): {train_r2:.4f}")

    sens = get_sobol2_fixed(Ci, mapping, exp_full)
    rows = []
    for key, val in sens.items():
        idxs = [int(s) for s in key[1:].split("_")]
        names = "+".join(VAR_COLS[i - 1] for i in idxs)
        rows.append((names, val))
    rows.sort(key=lambda r: -r[1])

    print("\n=== PDD 민감도 지수 (SEA 분산 기여율) ===")
    for names, val in rows:
        print(f"  {names:12s} {val:.4f}")

    # 교차 검증용: 변수별 SEA 단순 상관계수 (선형 PDD 계수와 방향/크기 비교용)
    print("\n=== 변수별 SEA 단순 상관계수 (교차검증용) ===")
    corrs = [(name, np.corrcoef(X_raw[i], Y)[0, 1]) for i, name in enumerate(VAR_COLS)]
    corrs.sort(key=lambda t: -abs(t[1]))
    for name, r in corrs:
        print(f"  {name:5s} r={r:+.3f}")


if __name__ == "__main__":
    main()
