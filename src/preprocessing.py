import nibabel as nib
import numpy as np
from scipy.ndimage import zoom, gaussian_filter
from src.config import IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH

def normalize_volume(volume):
    """Görüntüyü percentile tabanlı normalize eder."""
    p1, p99 = np.percentile(volume[volume > 0], [1, 99])
    volume = np.clip(volume, p1, p99)
    volume = (volume - p1) / (p99 - p1 + 1e-8)
    return volume

def enhance_contrast(volume):
    """Histogram eşitleme ile kontrastı artırır."""
    volume = volume - np.min(volume)
    if np.max(volume) > 0:
        volume = volume / np.max(volume)
    volume = np.power(volume, 0.8)
    return volume

def process_scan(path):
    """NIfTI/IMG dosyasını okur ve işler."""
    try:
        image = nib.load(path)
        volume = image.get_fdata().astype(np.float32)
        
        # 4D ise 3D'ye çevir
        if volume.ndim == 4:
            volume = volume[:, :, :, 0]
        
        # Gürültü azaltma
        volume = gaussian_filter(volume, sigma=0.5)
        volume[volume < 0] = 0
        
        # Normalizasyon
        if np.sum(volume > 0) > 100:
            volume = normalize_volume(volume)
        else:
            max_val = np.max(volume)
            if max_val != 0:
                volume = volume / max_val
        
        # Kontrast iyileştirme
        volume = enhance_contrast(volume)
            
        # Boyutlandırma
        depth_factor = IMG_DEPTH / volume.shape[-1]
        width_factor = IMG_WIDTH / volume.shape[0]
        height_factor = IMG_HEIGHT / volume.shape[1]
        volume = zoom(volume, (width_factor, height_factor, depth_factor), order=2)
        
        return volume
        
    except Exception as e:
        print(f"Hata: {path} işlenemedi -> {e}")
        return np.zeros((IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH))

def process_scan_multi(paths):
    """Birden fazla MPR dosyasını birleştirerek işler.
    
    Her MPR ayrı işlenir ve sonuçların ortalaması alınır.
    Bu yaklaşım gürültüyü azaltır ve daha güvenilir sonuçlar verir.
    
    Args:
        paths: MPR dosya yollarının listesi
        
    Returns:
        Birleştirilmiş ve normalize edilmiş 3D volume
    """
    if not paths:
        return np.zeros((IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH))
    
    volumes = []
    for path in paths:
        vol = process_scan(path)
        if np.sum(vol) > 0:  # Boş olmayan volumeleri ekle
            volumes.append(vol)
    
    if not volumes:
        return np.zeros((IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH))
    
    # Ortalamasını al
    combined = np.mean(volumes, axis=0)
    print(f"   [+] {len(volumes)} MPR birleştirildi")
    return combined.astype(np.float32)