#!/usr/bin/env python3
"""
주인님을 위한 간단한 실행 스크립트
빠른 테스트를 위해 에포크 수를 줄인 버전
"""

from mnist_gan_with_timing import MNISTGANTrainer
import torch

def quick_test():
    """빠른 테스트 실행 (10 에포크)"""
    print("주인님, 빠른 테스트를 시작합니다! (10 에포크)")
    
    trainer = MNISTGANTrainer(
        batch_size=100,
        z_dim=100,
        lr=0.0002
    )
    
    trainer.train(n_epochs=10, save_interval=5)
    print("빠른 테스트 완료!")

def full_training():
    """전체 훈련 실행 (200 에포크)"""
    print("주인님, 전체 훈련을 시작합니다! (200 에포크)")
    
    trainer = MNISTGANTrainer(
        batch_size=100,
        z_dim=100,
        lr=0.0002
    )
    
    trainer.train(n_epochs=200, save_interval=50)
    print("전체 훈련 완료!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        full_training()
    else:
        quick_test()
