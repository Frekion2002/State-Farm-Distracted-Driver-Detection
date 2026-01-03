import os
import json
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss

# ==========================================
# Configuration
# ==========================================
class CONFIG:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 경로 설정
    DATA_DIR = "/home/yongjin/Deep_Learning_term_project/imgs"
    TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train")
    CSV_PATH = "/home/yongjin/Deep_Learning_term_project/csv_file/driver_imgs_list.csv"
    BBOX_JSON_PATH = os.path.join(DATA_DIR, "yolo_pose_bboxes.json") 

    # 모델 하이퍼파라미터
    MODEL_NAME = "resnet50"
    NUM_CLASSES = 10
    
    # 이미지 크기 설정
    # ResNet50 입력(224) + 관심 영역(ROI) 크롭(128)
    ORIG_IMAGE_SIZE = (224, 224)
    CROP_IMAGE_SIZE = (128, 128) 
    
    PRETRAINED = True
    BATCH_SIZE = 32
    NUM_EPOCHS = 15
    
    # 차등 Learning Rate: Backbone은 천천히, Classifier는 빠르게 학습
    BACKBONE_LR = 1e-5
    CLASSIFIER_LR = 1e-4
    WEIGHT_DECAY = 1e-3
    
    NUM_WORKERS = 8
    N_SPLITS = 5
    RANDOM_STATE = 42

# ==========================================
# 1. Data Preprocessing (전처리 과정)
# ==========================================
def get_data_transforms():
    """
    학습 데이터 증강 및 전처리 파이프라인 정의
    - 정규화(Normalize): ImageNet 통계량을 사용하여 수렴 속도 향상
    - 증강(Augmentation): 과적합 방지를 위해 밝기/대조/Affine 변환 적용
    """
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # 원본 이미지용 변환
    train_transform_orig = transforms.Compose([
        transforms.Resize(CONFIG.ORIG_IMAGE_SIZE),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    # Validation용 변환 (증강 없음)
    val_transform_orig = transforms.Compose([
        transforms.Resize(CONFIG.ORIG_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    # Crop 이미지용 변환
    train_transform_crop = transforms.Compose([
        transforms.Resize(CONFIG.CROP_IMAGE_SIZE),
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # 소폭의 증강
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    val_transform_crop = transforms.Compose([
        transforms.Resize(CONFIG.CROP_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    return (train_transform_orig, train_transform_crop), (val_transform_orig, val_transform_crop)


class StateFarmDataset(Dataset): 
    """
    운전자 이미지 및 주요 신체 부위(머리, 손) Crop 데이터셋
    - YOLO로 추출한 BBox 정보를 활용하여 관심 영역(ROI)을 별도로 추출
    """
    def __init__(self, df, img_dir, bbox_data, transform_orig, transform_crop):
        self.df = df
        self.img_dir = img_dir
        self.bbox_data = bbox_data
        self.transform_orig = transform_orig
        self.transform_crop = transform_crop
        
        # BBox 누락 시 사용할 0 텐서
        self.dummy_crop_tensor = torch.zeros(3, CONFIG.CROP_IMAGE_SIZE[0], CONFIG.CROP_IMAGE_SIZE[1])

    def __len__(self):
        return len(self.df)

    def _get_crop(self, image, box):
        # BBox 정보가 있으면 Crop 후 변환, 없으면 Dummy 반환
        if box is None:
            return self.dummy_crop_tensor
        try:
            cropped_img = image.crop(box)
            return self.transform_crop(cropped_img)
        except Exception:
            return self.dummy_crop_tensor

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row['classname'][1:])
        img_name = row['img']
        class_name = row['classname']
        img_path = os.path.join(self.img_dir, class_name, img_name)
        
        # BBox Key 조회
        relative_key = f"{class_name}/{img_name}" 
        boxes = self.bbox_data.get(relative_key, {"head": None, "l_hand": None, "r_hand": None})

        try:
            image = Image.open(img_path).convert("RGB")

            # 1. 전체 이미지 변환
            img_orig = self.transform_orig(image)
            # 2. 머리 Crop
            img_head = self._get_crop(image, boxes['head'])
            # 3. 손 Crop
            hand_box = boxes['l_hand'] if boxes['l_hand'] is not None else boxes['r_hand']
            img_hand = self._get_crop(image, hand_box)

            return (img_orig, img_head, img_hand), label
        
        except Exception:
            # 이미지 로드 실패 시 예외 처리
            return (torch.randn(3, *CONFIG.ORIG_IMAGE_SIZE), self.dummy_crop_tensor, self.dummy_crop_tensor), -1

# ==========================================
# 2. Model Optimization (모델 최적화 구조)
# ==========================================
class MultiCropResNet(nn.Module):
    """
    Multi-Stream Network Architecture:
    - 전역 특징(Global): ResNet50 (전체 이미지)
    - 지역 특징(Local): ResNet18 (머리) + ResNet18 (손)
    - 특징 결합(Concat) 후 최종 분류하여 미세한 행동 변화 감지 성능 극대화
    """
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights_50 = models.ResNet50_Weights.DEFAULT if pretrained else None
        weights_18 = models.ResNet18_Weights.DEFAULT if pretrained else None

        # Backbone 1: 원본 이미지
        self.backbone_orig = models.resnet50(weights=weights_50)
        num_ftrs_orig = self.backbone_orig.fc.in_features 
        self.backbone_orig.fc = nn.Identity() 

        # Backbone 2: Head Crop
        self.backbone_head = models.resnet18(weights=weights_18)
        num_ftrs_head = self.backbone_head.fc.in_features 
        self.backbone_head.fc = nn.Identity()

        # Backbone 3: Hand Crop
        self.backbone_hand = models.resnet18(weights=weights_18)
        num_ftrs_hand = self.backbone_hand.fc.in_features 
        self.backbone_hand.fc = nn.Identity()

        total_features = num_ftrs_orig + num_ftrs_head + num_ftrs_hand
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.8),
            nn.Linear(total_features, num_classes)
        )

    def forward(self, inputs):
        img_orig, img_head, img_hand = inputs
        
        ftrs_orig = self.backbone_orig(img_orig)
        ftrs_head = self.backbone_head(img_head)
        ftrs_hand = self.backbone_hand(img_hand)
        
        # Feature Concatenation
        combined_features = torch.cat([ftrs_orig, ftrs_head, ftrs_hand], dim=1)
        return self.classifier(combined_features)


# ==========================================
# 3. Training & Validation
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0
    
    for inputs_tuple, labels in tqdm(loader, desc="Training"):
        inputs = [inp.to(device) for inp in inputs_tuple]
        labels = labels.to(device)

        # 유효하지 않은 데이터 필터링
        valid_indices = (labels != -1)
        if not valid_indices.all():
            inputs = [inp[valid_indices] for inp in inputs]
            labels = labels[valid_indices]
            if inputs[0].size(0) == 0: continue

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = inputs[0].size(0)
        running_loss += loss.item() * batch_size
        _, preds = torch.max(outputs, 1)
        correct_preds += torch.sum(preds == labels.data)
        total_samples += batch_size
    
    if total_samples == 0: return 0.0, 0.0
    return running_loss / total_samples, (correct_preds.double() / total_samples).item()


def validate(model, loader, device):
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for inputs_tuple, labels in tqdm(loader, desc="Validating"):
            inputs = [inp.to(device) for inp in inputs_tuple]
            labels = labels.to(device)

            valid_indices = (labels != -1)
            if not valid_indices.all():
                inputs = [inp[valid_indices] for inp in inputs]
                labels = labels[valid_indices]
                if inputs[0].size(0) == 0: continue

            outputs = model(inputs)
            all_logits.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    if not all_logits: return 999.0, np.array([]), np.array([])
    
    all_logits = np.vstack(all_logits)
    all_labels = np.concatenate(all_labels)
    # LogLoss 계산
    probs = torch.softmax(torch.tensor(all_logits), dim=1).numpy()
    val_logloss = log_loss(all_labels, probs)
    
    return val_logloss, all_logits, all_labels


def main():
    print(f"Model: Multi-Crop ResNet")
    
    # 데이터 로드
    df = pd.read_csv(CONFIG.CSV_PATH)
    df['label'] = df['classname'].str[1:].astype(int)
    
    with open(CONFIG.BBOX_JSON_PATH, 'r') as f:
        bbox_data = json.load(f)

    # Transform 설정
    (train_transform_orig, train_transform_crop), (val_transform_orig, val_transform_crop) = get_data_transforms()

    # GroupKFold: 동일 운전자가 Train/Val에 섞이지 않도록 분리
    gkf = GroupKFold(n_splits=CONFIG.N_SPLITS)
    
    oof_logits = np.zeros((len(df), CONFIG.NUM_CLASSES))
    oof_labels = np.zeros(len(df))
    oof_mask = np.zeros(len(df), dtype=bool)
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df['label'], groups=df['subject'])):
        print(f"\n{'='*20} Fold {fold+1}/{CONFIG.N_SPLITS} {'='*20}")
        
        train_ds = StateFarmDataset(df.iloc[train_idx], CONFIG.TRAIN_IMG_DIR, bbox_data, train_transform_orig, train_transform_crop)
        val_ds = StateFarmDataset(df.iloc[val_idx], CONFIG.TRAIN_IMG_DIR, bbox_data, val_transform_orig, val_transform_crop)
        
        train_loader = DataLoader(train_ds, batch_size=CONFIG.BATCH_SIZE, shuffle=True, num_workers=CONFIG.NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=CONFIG.BATCH_SIZE, shuffle=False, num_workers=CONFIG.NUM_WORKERS, pin_memory=True)

        # 모델 초기화
        model = MultiCropResNet(num_classes=CONFIG.NUM_CLASSES, pretrained=CONFIG.PRETRAINED).to(CONFIG.DEVICE)
        criterion = nn.CrossEntropyLoss()

        # Optimizer 최적화: Backbone은 LR을 낮게, Classifier는 높게 설정
        optimizer = optim.AdamW([
            {'params': [p for n, p in model.named_parameters() if "backbone" in n], 'lr': CONFIG.BACKBONE_LR},
            {'params': [p for n, p in model.named_parameters() if "classifier" in n], 'lr': CONFIG.CLASSIFIER_LR}
        ], weight_decay=CONFIG.WEIGHT_DECAY)
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG.NUM_EPOCHS)

        best_loss = float('inf')
        patience, trigger = 3, 0 # Early Stopping 설정

        for epoch in range(CONFIG.NUM_EPOCHS):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, CONFIG.DEVICE)
            val_loss, val_logits, val_labels_arr = validate(model, val_loader, CONFIG.DEVICE)
            
            print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f}")
            scheduler.step()
            
            # Best Model 저장 및 Early Stopping 체크
            if val_loss < best_loss:
                best_loss = val_loss
                trigger = 0
                torch.save(model.state_dict(), f"checkpoints/fold{fold+1}_best.pth")
                if len(val_logits) > 0:
                    oof_logits[val_idx] = val_logits
                    oof_labels[val_idx] = val_labels_arr
                    oof_mask[val_idx] = True
            else:
                trigger += 1
                if trigger >= patience:
                    print("Early stopping triggered")
                    break
        
        del model, optimizer, train_loader, val_loader
        torch.cuda.empty_cache()

    # 전체 OOF Score 계산
    if oof_mask.sum() > 0:
        valid_idx = np.where(oof_mask)[0]
        final_logloss = log_loss(oof_labels[valid_idx], torch.softmax(torch.tensor(oof_logits[valid_idx]), dim=1).numpy())
        print(f"Overall CV LogLoss: {final_logloss:.4f}")

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    main()