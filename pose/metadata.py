#csv 파일에 들어갈 데이터의 형식지정, 데이터 추가 파일입니다.
import pandas as pd

def create_dataframe():
    return pd.DataFrame(columns=[
        'photo_id',
        'frame_id',
        'num_people',
        'gender_distribution',
        'pose_type'
    ])

def add_data(df, photo_id, frame_id, num_people, gender_distribution, pose_type):
    new_data = {
        "photo_id": photo_id,
        "frame_id": frame_id,
        "num_people": num_people,
        "gender_distribution": gender_distribution,
        "pose_type": pose_type
    }
    return pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)