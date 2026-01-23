import torch
import torchaudio
import numpy as np
import random

class AudioAugmentation:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
    def change_speed(self, waveform: torch.Tensor, speed_factor: float = 1.0) -> torch.Tensor:
        if speed_factor == 1.0:
            return waveform
            
        # Tính tần số lấy mẫu mới
        new_freq = int(self.sample_rate * speed_factor)

        device = waveform.device
        resampler = torchaudio.transforms.Resample(
            orig_freq=new_freq, 
            new_freq=self.sample_rate
        ).to(device) 
        
        return resampler(waveform)
    
    def add_noise(self, waveform: torch.Tensor, min_noise: float = 0.003, max_noise: float = 0.008) -> torch.Tensor:
        device = waveform.device
        noise_level = random.uniform(min_noise, max_noise)
        
        noise = torch.randn_like(waveform, device=device)
        
        # Tính năng lượng để scale noise hợp lý
        energy = torch.norm(waveform)
        noise_energy = torch.norm(noise)
        
        if noise_energy == 0:
            return waveform
            
        scaled_noise = noise * (energy / noise_energy) * noise_level
        return waveform + scaled_noise

    def time_masking(self, waveform: torch.Tensor, max_mask_pct: float = 0.1) -> torch.Tensor:
        # Handle both 1D and 2D tensors
        if waveform.dim() == 1:
            len_wave = waveform.shape[0]
            is_1d = True
        else:
            _, len_wave = waveform.shape
            is_1d = False
        
        # Độ dài đoạn cần che (tối đa 10% độ dài file)
        mask_len = int(len_wave * random.uniform(0.01, max_mask_pct))
        
        # Vị trí bắt đầu che
        start = random.randint(0, len_wave - mask_len)
        
        # Tạo bản copy để không ảnh hưởng dữ liệu gốc
        masked_waveform = waveform.clone()
        
        # Gán bằng 0 (làm câm) đoạn đó
        if is_1d:
            masked_waveform[start:start + mask_len] = 0
        else:
            masked_waveform[:, start:start + mask_len] = 0
        
        return masked_waveform

    def augment_batch(self, waveform: torch.Tensor, augment_prob: float = 0.5) -> torch.Tensor:
        aug_waveform = waveform.clone()
        
        # 1. Change Speed - DISABLED (too slow, makes training 10x slower!)
        # if random.random() < augment_prob:
        #     speed = random.uniform(0.9, 1.1)
        #     try:
        #         aug_waveform = self.change_speed(aug_waveform, speed)
        #     except: pass

        # 2. Add Noise (Energy-scaled)
        if random.random() < augment_prob:
            # Dùng range mặc định mới (0.003 - 0.008)
            aug_waveform = self.add_noise(aug_waveform)
            
        # 3. Waveform Masking (Thay cho SpecAugment)
        # Giúp model học được khi bị mất thông tin
        if random.random() < augment_prob:
            aug_waveform = self.time_masking(aug_waveform, max_mask_pct=0.1)
            
        # 4. Random Gain
        if random.random() < augment_prob:
            gain = random.uniform(0.8, 1.2)
            aug_waveform = aug_waveform * gain
            
        return aug_waveform