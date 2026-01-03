import pandas as pd
import os
import json
import torch
from ultralytics import YOLO
from tqdm import tqdm

# ==========================================
# 1. 환경 설정 (Configuration)
# ==========================================
# 기본 경로
BASE_DIR = "/home/yongjin/Deep_Learning_term_project"
IMG_DIR = os.path.join(BASE_DIR, "imgs")
TRAIN_DIR = os.path.join(IMG_DIR, "train")
TEST_DIR = os.path.join(IMG_DIR, "test")

# CSV 파일 경로
TRAIN_CSV_PATH = os.path.join(BASE_DIR, "csv_file/driver_imgs_list.csv")
TEST_CSV_PATH = os.path.join(BASE_DIR, "csv_file/sample_submission.csv")

# 저장될 JSON 파일 경로
SAVE_JSON_PATH = os.path.join(IMG_DIR, "yolo_pose_bboxes.json")

# 모델 및 Crop 설정
MODEL_PATH = 'yolov8n-pose.pt'
HEAD_KP_INDICES = [0, 1, 2, 3, 4] # 코, 눈, 귀
LEFT_HAND_INDEX = 9
RIGHT_HAND_INDEX = 10

# Crop 크기
CROP_SIZE_HEAD = 128
CROP_SIZE_HAND = 96


# ==========================================
# 2. ROI(관심 영역) 추출 함수 (논리적 근거)
# ==========================================
def get_bbox_from_keypoint(kp_data, index, crop_size, img_w, img_h):
    """
    손(Hand) 좌표를 중심으로 BBox 생성
    - 이유: 운전 중 손의 위치(핸들, 기어, 문자 등)가 행동 분류의 결정적 단서이기 때문
    """
    if kp_data.shape[0] <= index: return None
    x, y, conf = kp_data[index]
    if conf < 0.1: return None # 신뢰도 낮은 좌표 제외

    half = crop_size // 2
    x1, y1 = max(0, int(x - half)), max(0, int(y - half))
    x2, y2 = min(img_w, int(x + half)), min(img_h, int(y + half))
    return [x1, y1, x2, y2]

def get_head_bbox(kp_data, crop_size, img_w, img_h):
    """
    얼굴(Head) 좌표들의 중심을 계산하여 BBox 생성
    - 이유: 운전자의 시선 및 고개 방향(전방 주시 태만 여부)을 파악하기 위함
    """
    coords = []
    for i in HEAD_KP_INDICES:
        if kp_data.shape[0] > i and kp_data[i, 2] > 0.1:
            coords.append(kp_data[i, :2])
    
    if not coords: return None
    
    # 여러 포인트의 중심점 계산
    center_x, center_y = torch.mean(torch.stack(coords), dim=0)
    
    half = crop_size // 2
    x1, y1 = max(0, int(center_x - half)), max(0, int(center_y - half))
    x2, y2 = min(img_w, int(center_x + half)), min(img_h, int(center_y + half))
    return [x1, y1, x2, y2]


# ==========================================
# 3. 데이터 처리 메인 로직
# ==========================================
def process_images(model, image_list, root_dir, is_train=True):
    """
    이미지 리스트를 받아 YOLO Inference를 수행하고 BBox 결과를 반환하는 함수
    """
    results_dict = {}
    
    desc_text = "Processing Train Data" if is_train else "Processing Test Data"
    
    for item in tqdm(image_list, desc=desc_text):
        # 경로 및 Key 설정
        if is_train:
            # Train
            cls, fname = item # (classname, img)
            rel_path = f"{cls}/{fname}"
            full_path = os.path.join(root_dir, cls, fname)
        else:
            # Test
            fname = item
            rel_path = fname
            full_path = os.path.join(root_dir, fname)

        try:
            # YOLO 추론
            results = model(full_path, device=0, verbose=False)
            
            # 결과 파싱
            head_box, l_hand, r_hand = None, None, None
            
            if results and results[0].keypoints and len(results[0].keypoints.data) > 0:
                res = results[0]
                h, w = res.orig_shape
                kps = res.keypoints.data[0].cpu()

                head_box = get_head_bbox(kps, CROP_SIZE_HEAD, w, h)
                l_hand = get_bbox_from_keypoint(kps, LEFT_HAND_INDEX, CROP_SIZE_HAND, w, h)
                r_hand = get_bbox_from_keypoint(kps, RIGHT_HAND_INDEX, CROP_SIZE_HAND, w, h)

            results_dict[rel_path] = {
                "head": head_box,
                "l_hand": l_hand,
                "r_hand": r_hand
            }
            
        except Exception as e:
            # 이미지 깨짐 등 에러 발생 시 로그 출력 후 건너뜀
            print(f"\n[Error] {rel_path}: {e}")
            continue
            
    return results_dict

def main():
    print("통합 데이터 전처리 시작 (Train + Test)")

    # 1. 모델 로드
    model = YOLO(MODEL_PATH)

    # 2. 데이터셋 로드
    print("CSV 파일 로딩 중")
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)

    # 리스트 변환
    # Train
    train_list = list(zip(train_df['classname'], train_df['img']))
    # Test
    test_list = test_df['img'].tolist()

    print(f"Train Images: {len(train_list)}")
    print(f"Test Images: {len(test_list)}")

    # 3. 전처리 실행
    # Train Data 처리
    train_bboxes = process_images(model, train_list, TRAIN_DIR, is_train=True)
    
    # Test Data 처리
    test_bboxes = process_images(model, test_list, TEST_DIR, is_train=False)

    # 4. 병합 및 저장
    print("데이터 병합 중")
    # 두 딕셔너리 병합
    full_bbox_data = train_bboxes.copy()
    full_bbox_data.update(test_bboxes)

    print(f"총 {len(full_bbox_data)}개의 데이터(Train+Test)를 저장합니다.")
    print(f"저장 경로: {SAVE_JSON_PATH}")

    with open(SAVE_JSON_PATH, 'w') as f:
        json.dump(full_bbox_data, f)

    print("모든 전처리 완료. (yolo_pose_bboxes.json 생성됨)")

if __name__ == "__main__":
    main()