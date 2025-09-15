import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval

def _annotate_bars(ax, values):
    for i, v in enumerate(values):
        ax.text(i, v, str(v), ha='center', va='bottom', fontsize=9)

def parse_gender_cell(x):
    """
    gender_distribution 컬럼을 dict로 파싱.
    - 정상 JSON 문자열: {"female":0,"male":2}
    - 파이썬 dict 문자열: "{'male': 2, 'female': 0}"
    - 이미 dict인 경우: 그대로
    실패 시 {"male":0,"female":0}
    """
    if isinstance(x, dict):
        m = int(x.get("male", 0)); f = int(x.get("female", 0))
        return {"male": m, "female": f}
    if pd.isna(x):
        return {"male":0, "female":0}
    s = str(x).strip()
    try:
        # 우선 JSON 시도
        obj = json.loads(s)
        return {"male": int(obj.get("male",0)), "female": int(obj.get("female",0))}
    except Exception:
        try:
            # 파이썬 literal dict
            obj = literal_eval(s)
            return {"male": int(obj.get("male",0)), "female": int(obj.get("female",0))}
        except Exception:
            return {"male":0, "female":0}

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def main(csv_path, outdir):
    ensure_dir(outdir)
    # 1) 로드
    df = pd.read_csv(csv_path)
    required_cols = {"photo_id","frame_id","num_people","gender_distribution","pose_type"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV에 필수 컬럼이 없습니다: {missing}")

    # 2) gender_distribution 파싱 → male,female 확장
    gd = df["gender_distribution"].apply(parse_gender_cell)
    df["male"] = gd.apply(lambda d: d.get("male",0))
    df["female"] = gd.apply(lambda d: d.get("female",0))

    # === A) 인원수(+성비)별 최다 포즈 산출 ===
    # 그룹: (num_people, male, female)
    grp = df.groupby(["num_people", "male", "female", "pose_type"]).size().reset_index(name="count")
    # 각 (num_people, male, female) 그룹에서 count 최대 pose만 추출
    idx = grp.groupby(["num_people", "male", "female"])['count'].idxmax()
    top_pose_pf = grp.loc[idx].sort_values(["num_people", "male", "female"]).reset_index(drop=True)
    # 보기 좋은 라벨
    top_pose_pf["group_label"] = top_pose_pf.apply(lambda r: f"{int(r['num_people'])}p | {int(r['male'])}M-{int(r['female'])}F", axis=1)
    # 저장
    top_pose_pf.to_csv(os.path.join(outdir, "table_top_pose_by_people_gender.csv"), index=False)

    # 시각화: 막대(그룹별 최다 포즈의 빈도), 막대 라벨에 pose_type 표시
    plt.figure()
    ax = plt.gca()
    ax.bar(top_pose_pf["group_label"], top_pose_pf["count"]) 
    # 포즈 라벨을 각 막대 위에 함께 표기
    for i, (lbl, cnt, pose) in enumerate(zip(top_pose_pf["group_label"], top_pose_pf["count"], top_pose_pf["pose_type"])):
        ax.text(i, cnt, pose, ha='center', va='bottom', fontsize=9, rotation=0)
    plt.title("Top pose per (num_people, gender mix)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plot_top_pose_by_people_gender.png"))
    plt.close()

    # === B) 인원수만 기준으로 최다 포즈 (성비 무시) ===
    grp2 = df.groupby(["num_people", "pose_type"]).size().reset_index(name="count")
    idx2 = grp2.groupby(["num_people"])['count'].idxmax()
    top_pose_p = grp2.loc[idx2].sort_values(["num_people"]).reset_index(drop=True)
    top_pose_p["group_label"] = top_pose_p.apply(lambda r: f"{int(r['num_people'])}p", axis=1)
    top_pose_p.to_csv(os.path.join(outdir, "table_top_pose_by_people.csv"), index=False)

    plt.figure()
    ax = plt.gca()
    ax.bar(top_pose_p["group_label"], top_pose_p["count"]) 
    for i, (lbl, cnt, pose) in enumerate(zip(top_pose_p["group_label"], top_pose_p["count"], top_pose_p["pose_type"])):
        ax.text(i, cnt, pose, ha='center', va='bottom', fontsize=9, rotation=0)
    plt.title("Top pose per num_people")
    plt.xlabel("num_people")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plot_top_pose_by_people.png"))
    plt.close()

    # 3) 요약 테이블들
    pose_freq = df["pose_type"].value_counts().rename_axis("pose_type").reset_index(name="count")
    pose_gender = df.groupby("pose_type")[["male","female"]].sum().reset_index()
    people_hist = df["num_people"].value_counts().sort_index().reset_index()
    people_hist.columns = ["num_people","count"]
    ppl_by_pose = df.groupby("pose_type")["num_people"].mean().round(2).reset_index(name="avg_num_people")

    # 4) 저장용 정제 CSV
    cleaned_path = os.path.join(outdir, "metadata_cleaned.csv")
    df.to_csv(cleaned_path, index=False)

    # 5) 표도 같이 저장
    pose_freq.to_csv(os.path.join(outdir, "table_pose_freq.csv"), index=False)
    pose_gender.to_csv(os.path.join(outdir, "table_pose_gender.csv"), index=False)
    people_hist.to_csv(os.path.join(outdir, "table_people_hist.csv"), index=False)
    ppl_by_pose.to_csv(os.path.join(outdir, "table_avg_people_by_pose.csv"), index=False)

    # 6) 시각화 (주의: seaborn 금지, 색 지정 금지)
    # (1) 포즈별 빈도
    plt.figure()
    plt.bar(pose_freq["pose_type"], pose_freq["count"])
    plt.title("Pose frequency")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plot_pose_frequency.png"))
    plt.close()

    # (2) 포즈별 성비 스택막대
    plt.figure()
    x = range(len(pose_gender))
    plt.bar(x, pose_gender["male"], label="male")
    plt.bar(x, pose_gender["female"], bottom=pose_gender["male"], label="female")
    plt.xticks(x, pose_gender["pose_type"], rotation=30, ha="right")
    plt.title("Gender distribution by pose")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plot_gender_by_pose.png"))
    plt.close()

    # (3) 인원수 히스토그램(막대)
    plt.figure()
    plt.bar(people_hist["num_people"].astype(str), people_hist["count"])
    plt.title("Num people per frame")
    plt.xlabel("num_people")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plot_num_people_hist.png"))
    plt.close()

    # (4) 포즈별 평균 인원수
    plt.figure()
    plt.bar(ppl_by_pose["pose_type"], ppl_by_pose["avg_num_people"])
    plt.title("Average num_people by pose")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("avg num_people")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plot_avg_people_by_pose.png"))
    plt.close()

    # 7) 콘솔 요약
    print("=== SUMMARY ===")
    print("Total rows:", len(df))
    print("\nPose frequency:")
    print(pose_freq.head(20).to_string(index=False))
    print("\nGender by pose:")
    print(pose_gender.head(20).to_string(index=False))
    print("\nNum people histogram:")
    print(people_hist.to_string(index=False))
    print("\nAverage num_people by pose:")
    print(ppl_by_pose.head(20).to_string(index=False))

    print("\nTop pose per (num_people, gender mix):")
    print(top_pose_pf[["num_people","male","female","pose_type","count"]].to_string(index=False))

    print("\nTop pose per num_people:")
    print(top_pose_p[["num_people","pose_type","count"]].to_string(index=False))

    print(f"\nSaved cleaned CSV → {cleaned_path}")
    print(f"Plots & tables saved under → {outdir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to metadata.csv")
    ap.add_argument("--outdir", default="analysis_outputs")
    args = ap.parse_args()
    main(args.csv, args.outdir)