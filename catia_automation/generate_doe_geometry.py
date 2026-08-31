"""
generate_doe_geometry.py

목적: doe_points.csv의 각 DOE 점마다 CATIA 파라미터를 세팅하고
      bumper beam / crash box 3 형상을 각각 STEP으로 export.
      (docs/GUIDELINE.md 파이프라인 2단계 "형상 자동 생성(카티아)")

전제 (케드 자동화 준비 세션에서 pycatia로 검증한 내용 반영):
  - CATIA가 이미 실행 중이고 assembly.CATProduct(또는 최소 bumper beam.CATPart,
    crash box 3.CATPart)가 열려 있어야 함
  - CATIA 파라미터 이름은 CSV 컬럼명과 다르게 "DOE_" 접두사가 붙어 있음
    (예: CSV의 t_beam -> CATIA 파라미터 DOE_t_beam)
  - beam 쪽 파라미터(t_beam, beam_radius, beam_height)는 bumper beam.CATPart에,
    box 쪽 파라미터(t_box, box_length, rib_offset, fillet_center_offset)는
    crash box 3.CATPart에 있음 -> 두 파트를 각각 update/export
  - Symmetry of Part3_1.CATPart(대칭 복사본)는 파라미터가 없는 Assemble 결과물이라
    이 스크립트에서 다루지 않음 -> 대칭 인스턴스는 Abaqus 어셈블리 단계에서 처리
  - doe_points.csv의 마지막 2열(yield_strength_delta, thickness_tolerance)은
    noise 변수라 CATIA 형상에는 반영하지 않음 (yield_strength_delta는 Abaqus 재질
    카드로, thickness_tolerance는 형상 vs 쉘 두께 반영 여부가 아직 미확정이라 일단 제외)
  - Part.update()는 극단적인 DOE 점에서 실패할 수 있음(beam_height LB 근처에서 실제
    확인됨) -> 실패하면 해당 점은 건너뛰고 로그만 남김
  - 원본 CATPart 마스터 파일을 계속 덮어쓰지 않도록 이 스크립트에서는 doc.save()를
    호출하지 않음 (메모리상 업데이트된 형상만 export)

export_data(file_name, "stp", overwrite=True)는 케드 자동화 준비 세션에서 실제로
검증됨 (200,690 bytes STEP, ISO-10303-21/AP203 정상 생성 확인, master CATPart는
save()를 안 부르니 안 건드림 확인됨). overwrite=True를 안 주면 같은 이름 파일이
이미 있을 때 덮어쓰기가 안 될 수 있어서 DOE 루프에서는 반드시 필요.

실행 전: CSV_PATH, OUTPUT_DIR을 실제 경로로 수정하고 OUTPUT_DIR을 미리 만들어둘 것.
"""

import csv
from pathlib import Path

from pycatia import catia

CSV_PATH = Path(r"C:\Users\PC\Desktop\bumper\bumper-PDD-design\doe_points.csv")  # TODO: 실제 경로
OUTPUT_DIR = Path(r"C:\Users\PC\Desktop\bumper\export")  # TODO: 실제 경로

BEAM_COLS = ["t_beam", "beam_radius", "beam_height"]
BOX_COLS = ["t_box", "box_length", "rib_offset", "fillet_center_offset"]


def find_doc(docs, name_substr):
    for i in range(1, docs.count + 1):
        d = docs.item(i)
        if name_substr in d.name:
            return d
    raise ValueError(f"문서를 찾을 수 없음: {name_substr}")


def find_param(part, suffix):
    for i in range(1, part.parameters.count + 1):
        p = part.parameters.item(i)
        if p.name.endswith(suffix):
            return p
    raise ValueError(f"파라미터를 찾을 수 없음: {suffix}")


def set_params(part, cols, row):
    for col in cols:
        param = find_param(part, f"DOE_{col}")
        param.value = float(row[col])


def main():
    caa = catia()
    caa.visible = True
    docs = caa.documents

    beam_doc = find_doc(docs, "bumper beam")
    box_doc = find_doc(docs, "crash box 3")
    beam_part = beam_doc.part
    box_part = box_doc.part

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    failed = []
    for idx, row in enumerate(rows, start=1):
        try:
            set_params(beam_part, BEAM_COLS, row)
            beam_part.update()

            set_params(box_part, BOX_COLS, row)
            box_part.update()
        except Exception as e:
            print(f"[SKIP] design_{idx:04d}: update 실패 ({e})")
            failed.append(idx)
            continue

        beam_path = OUTPUT_DIR / f"design_{idx:04d}_beam.stp"
        box_path = OUTPUT_DIR / f"design_{idx:04d}_box.stp"
        beam_doc.export_data(str(beam_path), "stp", overwrite=True)
        box_doc.export_data(str(box_path), "stp", overwrite=True)

        print(f"[OK] design_{idx:04d} -> {beam_path.name}, {box_path.name}")

    print(f"\n완료: {len(rows) - len(failed)}/{len(rows)}개 성공, 실패 {len(failed)}개 {failed}")


if __name__ == "__main__":
    main()
