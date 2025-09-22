#!/usr/bin/env python3
"""
PyTorch MNIST GAN with Timing Measurements
주인님을 위한 시간 측정 기능이 포함된 MNIST GAN 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import time
import os
from datetime import datetime


class Generator(nn.Module):
    """생성자 네트워크"""
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(g_input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)
    
    def forward(self, x): 
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))


class Discriminator(nn.Module):
    """판별자 네트워크"""
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)
    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))


class MNISTGANTrainer:
    """MNIST GAN 훈련 클래스"""
    
    def __init__(self, batch_size=100, z_dim=100, lr=0.0002, device=None):
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.lr = lr
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 시간 측정을 위한 변수들
        self.training_times = []
        self.epoch_times = []
        self.total_start_time = None
        
        print(f"주인님, 사용 중인 디바이스: {self.device}")
        
        # 데이터 로더 설정
        self._setup_data_loaders()
        
        # 네트워크 초기화
        self._setup_networks()
        
        # 손실 함수 및 옵티마이저 설정
        self._setup_training_components()
        
        # 샘플 저장 디렉토리 생성
        os.makedirs('./samples', exist_ok=True)
    
    def _setup_data_loaders(self):
        """데이터 로더 설정"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        
        train_dataset = datasets.MNIST(root='./mnist_data/', train=True, 
                                     transform=transform, download=True)
        test_dataset = datasets.MNIST(root='./mnist_data/', train=False, 
                                    transform=transform, download=False)
        
        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # MNIST 이미지 차원
        self.mnist_dim = train_dataset.train_data.size(1) * train_dataset.train_data.size(2)
        print(f"MNIST 이미지 차원: {self.mnist_dim}")
    
    def _setup_networks(self):
        """네트워크 초기화"""
        self.G = Generator(g_input_dim=self.z_dim, g_output_dim=self.mnist_dim).to(self.device)
        self.D = Discriminator(self.mnist_dim).to(self.device)
        
        print("생성자 네트워크:")
        print(self.G)
        print("\n판별자 네트워크:")
        print(self.D)
    
    def _setup_training_components(self):
        """훈련 구성 요소 설정"""
        self.criterion = nn.BCELoss()
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.lr)
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=self.lr)
    
    def train_discriminator(self, x):
        """판별자 훈련"""
        self.D.zero_grad()
        
        # 실제 데이터로 훈련
        x_real = x.view(-1, self.mnist_dim)
        y_real = torch.ones(x_real.size(0), 1)
        x_real, y_real = Variable(x_real.to(self.device)), Variable(y_real.to(self.device))
        
        D_output = self.D(x_real)
        D_real_loss = self.criterion(D_output, y_real)
        
        # 가짜 데이터로 훈련
        z = Variable(torch.randn(x_real.size(0), self.z_dim).to(self.device))
        x_fake = self.G(z)
        y_fake = Variable(torch.zeros(x_real.size(0), 1).to(self.device))
        
        D_output = self.D(x_fake)
        D_fake_loss = self.criterion(D_output, y_fake)
        
        # 역전파 및 최적화
        D_loss = D_real_loss + D_fake_loss
        D_loss.backward()
        self.D_optimizer.step()
        
        return D_loss.data.item()
    
    def train_generator(self, batch_size):
        """생성자 훈련"""
        self.G.zero_grad()
        
        z = Variable(torch.randn(batch_size, self.z_dim).to(self.device))
        y = Variable(torch.ones(batch_size, 1).to(self.device))
        
        G_output = self.G(z)
        D_output = self.D(G_output)
        G_loss = self.criterion(D_output, y)
        
        # 역전파 및 최적화
        G_loss.backward()
        self.G_optimizer.step()
        
        return G_loss.data.item()
    
    def train(self, n_epochs=200, save_interval=50):
        """GAN 훈련"""
        print(f"\n주인님, {n_epochs} 에포크 동안 훈련을 시작합니다...")
        print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.total_start_time = time.time()
        
        for epoch in range(1, n_epochs + 1):
            epoch_start_time = time.time()
            
            D_losses, G_losses = [], []
            
            for batch_idx, (x, _) in enumerate(self.train_loader):
                # 판별자 훈련
                d_loss = self.train_discriminator(x)
                D_losses.append(d_loss)
                
                # 생성자 훈련
                g_loss = self.train_generator(x.size(0))
                G_losses.append(g_loss)
            
            epoch_time = time.time() - epoch_start_time
            self.epoch_times.append(epoch_time)
            
            # 에포크 결과 출력
            avg_d_loss = torch.mean(torch.FloatTensor(D_losses))
            avg_g_loss = torch.mean(torch.FloatTensor(G_losses))
            
            print(f'[{epoch}/{n_epochs}]: loss_d: {avg_d_loss:.3f}, loss_g: {avg_g_loss:.3f}, '
                  f'시간: {epoch_time:.2f}초')
            
            # 주기적으로 샘플 이미지 저장
            if epoch % save_interval == 0:
                self.generate_samples(epoch)
        
        total_time = time.time() - self.total_start_time
        self.print_timing_summary(total_time, n_epochs)
        
        # 최종 샘플 생성
        self.generate_samples('final')
    
    def generate_samples(self, epoch):
        """샘플 이미지 생성 및 저장"""
        with torch.no_grad():
            test_z = Variable(torch.randn(self.batch_size, self.z_dim).to(self.device))
            generated = self.G(test_z)
            
            filename = f'./samples/sample_epoch_{epoch}.png'
            save_image(generated.view(generated.size(0), 1, 28, 28), filename)
            print(f"샘플 이미지 저장: {filename}")
    
    def print_timing_summary(self, total_time, n_epochs):
        """시간 측정 결과 요약 출력"""
        print("\n" + "="*60)
        print("주인님을 위한 훈련 시간 요약")
        print("="*60)
        print(f"총 훈련 시간: {total_time:.2f}초 ({total_time/60:.2f}분)")
        print(f"평균 에포크 시간: {sum(self.epoch_times)/len(self.epoch_times):.2f}초")
        print(f"최빠른 에포크: {min(self.epoch_times):.2f}초")
        print(f"최느린 에포크: {max(self.epoch_times):.2f}초")
        print(f"에포크당 평균 속도: {total_time/n_epochs:.2f}초/에포크")
        print(f"완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)


def main():
    """메인 함수"""
    print("주인님, PyTorch MNIST GAN 훈련을 시작합니다!")
    
    # 하이퍼파라미터 설정
    config = {
        'batch_size': 100,
        'z_dim': 100,
        'lr': 0.0002,
        'n_epochs': 200,
        'save_interval': 50
    }
    
    print(f"설정: {config}")
    
    # 트레이너 초기화 및 훈련 시작
    trainer = MNISTGANTrainer(
        batch_size=config['batch_size'],
        z_dim=config['z_dim'],
        lr=config['lr']
    )
    
    trainer.train(
        n_epochs=config['n_epochs'],
        save_interval=config['save_interval']
    )
    
    print("주인님, 훈련이 완료되었습니다!")


if __name__ == "__main__":
    main()
