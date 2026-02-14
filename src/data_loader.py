import pandas as pd
import numpy as np
import os
import re
from src.preprocessing import process_scan, process_scan_multi
from src.config import RAW_IMG_DIRS

class DataLoader:
    """OASIS-2 veri seti yükleyici."""
    
    def __init__(self, metadata_path):
        self.metadata_path = metadata_path
        self.df = None
        
    def load_and_clean_csv(self):
        """CSV dosyasını okur ve temizler."""
        if not os.path.exists(self.metadata_path):
            print(f"HATA: CSV bulunamadı: {self.metadata_path}")
            return None

        df = pd.read_csv(self.metadata_path)
        
        # Sayısal sütunları düzelt
        numeric_cols = ['Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF', 'CDR']
        for col in numeric_cols:
            if col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Hedef değişken: CDR >= 0.5 ise Demans (1), değilse Sağlıklı (0)
        # NOT: 'Converted' grubundaki hastaların sağlıklı oldukları dönemleri (CDR=0)
        # yanlışlıkla hasta olarak etiketlememek için Group yerine CDR kullanıyoruz.
        df['Target'] = df['CDR'].apply(lambda x: 1 if x >= 0.5 else 0)
        
        # Cinsiyet: M -> 1, F -> 0
        df['Gender_Num'] = df['M/F'].apply(lambda x: 1 if x == 'M' else 0)
        
        # Eksik verileri medyan ile doldur
        for col in ['MMSE', 'SES', 'EDUC', 'eTIV', 'nWBV', 'CDR']:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        df = df.dropna(subset=['MRI ID'])
        self.df = df
        return df

    def _find_all_mpr_files(self, base_path):
        """RAW klasöründeki tüm MPR dosyalarını bulur (mpr-1'den son MPR'a kadar).
        
        Args:
            base_path: RAW klasör yolu
            
        Returns:
            Sıralı MPR dosya yollarının listesi
        """
        if not os.path.exists(base_path):
            return []
        
        mpr_files = []
        pattern = re.compile(r'mpr-(\d+)\.(nifti\.)?img$', re.IGNORECASE)
        
        for fname in os.listdir(base_path):
            match = pattern.match(fname)
            if match:
                mpr_num = int(match.group(1))
                full_path = os.path.join(base_path, fname)
                mpr_files.append((mpr_num, full_path))
        
        # MPR numarasına göre sırala ve sadece yolları döndür
        mpr_files.sort(key=lambda x: x[0])
        return [path for _, path in mpr_files]

    def load_matched_data(self):
        """MR görüntüleri ile eşleşen verileri yükler (tüm MPR'lar birleştirilerek)."""
        if self.df is None:
            self.load_and_clean_csv()
            
        X_img, X_tab, y = [], [], []
        feature_cols = ['Age', 'Gender_Num', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV']
        
        print(f"[*] {len(self.df)} kayıt taranıyor...")
        count = 0
        
        for _, row in self.df.iterrows():
            mri_id = row['MRI ID']
            
            # Tüm MPR dosyalarını bul
            mpr_paths = []
            for raw_dir in RAW_IMG_DIRS:
                base_path = os.path.join(raw_dir, mri_id, 'RAW')
                mpr_paths = self._find_all_mpr_files(base_path)
                if mpr_paths:
                    break
            
            if mpr_paths:
                try:
                    # Tüm MPR'ları birleştirerek işle
                    vol = process_scan_multi(mpr_paths)
                    X_img.append(vol)
                    X_tab.append(row[feature_cols].values.astype('float32'))
                    y.append(row['Target'])
                    
                    count += 1
                    if count % 10 == 0:
                        print(f"   -> {count} görüntü yüklendi...")
                except Exception as e:
                    print(f"[!] Yüklenemedi: {mri_id} -> {e}")
            
        print(f"[OK] Toplam: {count} veri yüklendi")
        return np.array(X_img), np.array(X_tab), np.array(y)