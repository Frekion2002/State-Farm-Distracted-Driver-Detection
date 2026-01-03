import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

class CONFIG:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 경로 설정
    DATA_DIR = "/home/yongjin/Deep_Learning_term_project/imgs"
    TEST_IMG_DIR = os.path.join(DATA_DIR, "test")
    BBOX_JSON_PATH = os.path.join(DATA_DIR, "yolo_pose_bboxes.json")
    
    CHECKPOINT_DIR = "/home/yongjin/Deep_Learning_term_project/code/ResNet2/checkpoints"
    SAMPLE_SUBMISSION = "/home/yongjin/Deep_Learning_term_project/csv_file/sample_submission.csv"
    
    NUM_CLASSES = 10
    BATCH_SIZE = 32
    NUM_WORKERS = 8
    K_NEIGHBORS = 50  # KNN 이웃 수

class TestDataset(Dataset):
    """BBox 정보를 이용해 원본, 머리, 손 이미지를 반환하는 데이터셋"""
    def __init__(self, img_names, img_dir, bbox_data, transform_orig, transform_crop):
        self.img_names = img_names
        self.img_dir = img_dir
        self.bbox_data = bbox_data
        self.transform_orig = transform_orig
        self.transform_crop = transform_crop
        self.dummy_crop = torch.zeros(3, 128, 128)
    
    def __len__(self):
        return len(self.img_names)
    
    def _get_crop(self, image, box):
        if box is None: return self.dummy_crop
        try: return self.transform_crop(image.crop(box))
        except: return self.dummy_crop
    
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
            boxes = self.bbox_data.get(img_name, {})
            
            img_orig = self.transform_orig(image)
            img_head = self._get_crop(image, boxes.get('head'))
            img_hand = self._get_crop(image, boxes.get('l_hand') or boxes.get('r_hand'))
            
            return (img_orig, img_head, img_hand), img_name
        except:
            return (torch.zeros(3, 224, 224), self.dummy_crop, self.dummy_crop), img_name

class MultiCropResNet(nn.Module):
    """전신+머리+손 특징을 결합하는 모델"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone_orig = models.resnet50(weights=None)
        self.backbone_orig.fc = nn.Identity()
        
        self.backbone_head = models.resnet18(weights=None)
        self.backbone_head.fc = nn.Identity()
        
        self.backbone_hand = models.resnet18(weights=None)
        self.backbone_hand.fc = nn.Identity()
        
        # FC Layer 입력 차원 계산
        total_features = 2048 + 512 + 512
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.8),
            nn.Linear(total_features, num_classes)
        )
    
    def forward(self, inputs, return_features=False):
        img_orig, img_head, img_hand = inputs
        ftrs = torch.cat([
            self.backbone_orig(img_orig),
            self.backbone_head(img_head),
            self.backbone_hand(img_hand)
        ], dim=1)
        
        if return_features: return ftrs
        return self.classifier(ftrs)

def extract_features_and_predictions(model, dataloader, device):
    """KNN을 위한 Feature Vector 및 예측값 추출"""
    model.eval()
    feats, preds, names = [], [], []
    
    with torch.no_grad():
        for inputs, img_names in tqdm(dataloader, desc="Extracting Features"):
            inputs = [x.to(device) for x in inputs]
            
            # Feature
            feats.append(model(inputs, return_features=True).cpu().numpy())
            # Predictions
            preds.append(torch.softmax(model(inputs, return_features=False), dim=1).cpu().numpy())
            names.extend(img_names)
            
    return np.vstack(feats), np.vstack(preds), names

def knn_smoothing(predictions, features, k=50):
    """
    핵심 로직: Feature 유사도를 기반으로 예측 확률 분포를 보정(Smoothing)
    """
    print(f"\n[KNN] Smoothing Start (k={k})...")

    # 1. Feature 정규화 (Cosine Similarity 효과)
    features = normalize(features, axis=1, norm='l2')

    # 2. 최근접 이웃 탐색
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean', n_jobs=-1)
    nbrs.fit(features)
    distances, indices = nbrs.kneighbors(features)

    # 3. 거리 기반 가중치
    weights = np.exp(-distances * 5)
    weights = weights / weights.sum(axis=1, keepdims=True)
    
    # 4. 이웃들의 예측값을 가중 기하 평균으로 반영
    smoothed_preds = np.zeros_like(predictions)
    for i in tqdm(range(len(predictions)), desc="KNN Averaging"):
        neighbor_preds = predictions[indices[i]]
        
        # log 공간에서 평균은 기하 평균 효과
        log_preds = np.log(neighbor_preds + 1e-15) 
        weighted_log = (log_preds * weights[i][:, None]).sum(axis=0)
        smoothed_preds[i] = np.exp(weighted_log)
    
    # 확률 합 1로 정규화
    return smoothed_preds / smoothed_preds.sum(axis=1, keepdims=True)

def main():
    # 1. 설정 및 데이터 로드
    print("Loading Data")
    with open(CONFIG.BBOX_JSON_PATH, 'r') as f:
        bbox_data = json.load(f)
    test_df = pd.read_csv(CONFIG.SAMPLE_SUBMISSION)
    
    # 전처리
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trans_orig = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])
    trans_crop = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(), norm])
    
    dataset = TestDataset(test_df['img'].tolist(), CONFIG.TEST_IMG_DIR, bbox_data, trans_orig, trans_crop)
    loader = DataLoader(dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False, num_workers=CONFIG.NUM_WORKERS, pin_memory=True)

    # 2. 모델 로드 및 Feature 추출
    models_list = []
    # Fold 모델 로드
    for i in range(1, 6):
        p = os.path.join(CONFIG.CHECKPOINT_DIR, f"fold{i}_dropout08_simple_best.pth")
        if os.path.exists(p):
            m = MultiCropResNet(CONFIG.NUM_CLASSES).to(CONFIG.DEVICE)
            m.load_state_dict(torch.load(p, map_location=CONFIG.DEVICE))
            m.eval()
            models_list.append(m)
            print(f"Loaded: fold{i}")

    if not models_list:
        raise FileNotFoundError("체크포인트 파일이 없습니다. 경로를 확인해주세요.")

    # 모든 모델의 Feature/Prediction 평균 계산
    agg_preds, agg_feats = [], []
    img_names = []
    
    print("\nExtracting Features from Models")
    for model in models_list:
        f, p, names = extract_features_and_predictions(model, loader, CONFIG.DEVICE)
        agg_feats.append(f)
        agg_preds.append(p)
        img_names = names # 이름은 동일하므로 덮어쓰기

    # 모델 간 평균
    final_feats = np.mean(agg_feats, axis=0)
    final_preds = np.mean(agg_preds, axis=0)

    # 3. KNN Post-Processing
    knn_preds = knn_smoothing(final_preds, final_feats, k=CONFIG.K_NEIGHBORS)

    # 4. 결과 저장
    output_path = CONFIG.SAMPLE_SUBMISSION.replace('sample_submission.csv', 'submission_knn.csv')
    
    submission = pd.DataFrame(knn_preds, columns=[f'c{i}' for i in range(10)])
    submission.insert(0, 'img', img_names)
    submission.to_csv(output_path, index=False)
    
    print(f"\nKNN Submission Saved: {output_path}")

if __name__ == "__main__":
    main()