# 주인님을 위한 PyTorch MNIST GAN 사용 가이드

## 설치 방법

```bash
# 의존성 설치
pip install -r requirements.txt
```

## 실행 방법

### 1. 빠른 테스트 (10 에포크)
```bash
python run_gan.py
```

### 2. 전체 훈련 (200 에포크)
```bash
python run_gan.py full
```

### 3. 메인 스크립트 직접 실행
```bash
python mnist_gan_with_timing.py
```

## 출력 결과

### 시간 측정 정보
- 총 훈련 시간
- 에포크별 시간
- 평균/최빠른/최느린 에포크 시간
- 완료 시간

### 생성된 파일
- `samples/sample_epoch_*.png`: 주기적으로 생성되는 샘플 이미지
- `samples/sample_final.png`: 최종 생성된 샘플 이미지

## 하이퍼파라미터 수정

`mnist_gan_with_timing.py`의 `main()` 함수에서 다음 값들을 수정할 수 있습니다:

```python
config = {
    'batch_size': 100,      # 배치 크기
    'z_dim': 100,          # 잠재 벡터 차원
    'lr': 0.0002,          # 학습률
    'n_epochs': 200,       # 에포크 수
    'save_interval': 50    # 샘플 저장 간격
}
```

## 주요 기능

1. **시간 측정**: 전체 훈련 시간과 에포크별 시간을 정확히 측정
2. **자동 디바이스 감지**: CUDA 사용 가능 시 자동으로 GPU 사용
3. **주기적 샘플 저장**: 훈련 과정에서 생성된 이미지를 주기적으로 저장
4. **구조화된 코드**: 클래스 기반으로 깔끔하게 구조화
5. **한국어 출력**: 주인님을 위한 친근한 한국어 메시지
