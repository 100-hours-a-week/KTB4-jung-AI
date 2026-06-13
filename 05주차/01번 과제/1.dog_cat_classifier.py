"""
ResNet50 기반 개/고양이/기타 분류기

[개발 과정 요약]

1. 요청: ResNet 모델을 불러와서 개와 고양이를 분류하는 기능 구현
   - torchvision의 pretrained ResNet50(ImageNet1K_V2)을 사용
   - 별도 학습 없이 ImageNet 클래스 인덱스(개: 151~268, 고양이: 281~285)의
     확률을 합산해 비교하는 zero-shot 방식으로 구현
   - classifier.py (모델 클래스) + main.py (CLI) + train.py (파인튜닝) 3파일로 구성

2. 요청: dog/cat 폴더의 이미지 파일을 순회하는 코드로 전환
   - data/dog/, data/cat/ 폴더를 순회하며 각 이미지에 대해 추론 수행
   - 폴더명을 정답 레이블로 사용해 [O]/[X] 및 정확도 출력

3. 요청: classifier.py와 main.py를 하나의 파일(1.dog_cat_classifier.py)로 통합
   - 두 파일을 단일 파일로 합치고 불필요한 추상화 제거

4. 요청: 개/고양이 외 ETC(기타) 확률을 추가해 3개 선택지로 확장
   - etc_score = 1 - dog_score - cat_score 로 계산
   - 단, ETC가 878개 클래스의 합이라 항상 지배적으로 나오는 문제 발생
   - 해결: dog + cat 합산이 0.1 미만일 때만 ETC 판정, 이상이면 dog/cat 간 정규화 비교

5. 요청: 순회 폴더에 etc 추가 (기타로 분류되면 정답 처리)
   - data/etc/ 폴더 추가, ETC로 분류 시 정답으로 집계
"""

import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

# ImageNet 클래스 인덱스: 개(151~268), 고양이(281~285)
_DOG_INDICES = set(range(151, 269))
_CAT_INDICES = set(range(281, 286))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class DogCatClassifier:
    def __init__(self, device: str | None = None):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image: str | Path | Image.Image) -> dict:
        if isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            img = Image.open(image).convert("RGB")

        tensor = TRANSFORM(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1).squeeze()

        dog_score = sum(probs[i].item() for i in _DOG_INDICES)
        cat_score = sum(probs[i].item() for i in _CAT_INDICES)
        combined = dog_score + cat_score
        etc_score = 1.0 - combined

        # dog/cat 합산 확률이 임계값 미만이면 ETC로 판정
        if combined < 0.1:
            label = "etc"
            confidence = etc_score
        elif dog_score >= cat_score:
            label = "dog"
            confidence = dog_score / combined
        else:
            label = "cat"
            confidence = cat_score / combined

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "dog": round(dog_score, 4),
            "cat": round(cat_score, 4),
            "etc": round(etc_score, 4),
        }


def run_folder(classifier: DogCatClassifier, folder: Path, true_label: str) -> tuple[int, int]:
    images = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not images:
        print(f"  [WARN] No images found in {folder}")
        return 0, 0

    correct = 0
    for img_path in images:
        result = classifier.predict(img_path)
        mark = "O" if result["label"] == true_label else "X"
        conf = result["confidence"] * 100
        print(f"  [{mark}] {img_path.name:40s} -> {result['label'].upper():3s}  ({conf:.1f}%)  [dog={result['dog']:.3f}, cat={result['cat']:.3f}, etc={result['etc']:.3f}]")
        if result["label"] == true_label:
            correct += 1

    return correct, len(images)


def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")

    dog_dir = data_dir / "dog"
    cat_dir = data_dir / "cat"

    if not dog_dir.exists() and not cat_dir.exists():
        print(f"Error: '{data_dir}/dog' 또는 '{data_dir}/cat' 폴더가 없습니다.")
        print("Usage: python 1.dog_cat_classifier.py [data_dir]")
        sys.exit(1)

    classifier = DogCatClassifier()
    print(f"ResNet50 classifier ready (device: {classifier.device})\n")

    total_correct, total_count = 0, 0

    for folder, label in [(dog_dir, "dog"), (cat_dir, "cat"), (data_dir / "etc", "etc")]:
        if not folder.exists():
            print(f"[SKIP] '{folder}' 폴더 없음\n")
            continue
        print(f"[{label.upper()}] {folder}")
        correct, count = run_folder(classifier, folder, label)
        total_correct += correct
        total_count += count
        if count:
            print(f"  -> {correct}/{count} 정답 ({correct/count*100:.1f}%)\n")

    if total_count:
        print(f"전체 정확도: {total_correct}/{total_count} ({total_correct/total_count*100:.1f}%)")


if __name__ == "__main__":
    main()
