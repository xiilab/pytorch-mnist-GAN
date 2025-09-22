#!/usr/bin/env python3
"""
주인님을 위한 초고속 테스트 스크립트
매우 작은 설정으로 빠르게 테스트
"""

from mnist_gan_with_timing import MNISTGANTrainer
import torch

def ultra_quick_test():
    """초고속 테스트 (3 에포크, 작은 배치)"""
    print("주인님, 초고속 테스트를 시작합니다! (3 에포크, 작은 배치)")
    
    trainer = MNISTGANTrainer(
        batch_size=32,  # 작은 배치 크기
        z_dim=50,       # 작은 잠재 벡터
        lr=0.001        # 높은 학습률
    )
    
    trainer.train(n_epochs=3, save_interval=1)
    print("초고속 테스트 완료!")

if __name__ == "__main__":
    ultra_quick_test()
