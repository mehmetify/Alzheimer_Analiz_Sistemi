import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.config import METADATA_PATH, TEST_SAMPLE_DIR, EPOCHS, BATCH_SIZE
from src.data_loader import DataLoader
from src.models import build_3d_cnn, get_ml_model, HybridModel
from src.analysis import plot_correlation_matrix, compare_patient_to_population, generate_discussion
from src.preprocessing import process_scan

def main():
    print("\n🧠 ALZHEIMER ANALİZ SİSTEMİ BAŞLATILIYOR...\n")

    # --- 1. VERİ YÜKLEME ---
    loader = DataLoader(METADATA_PATH)
    
    # CSV Analizi
    df = loader.load_and_clean_csv()
    if df is not None:
        print("📊 Korelasyon Matrisi Çiziliyor...")
        plot_correlation_matrix(df)

    # Veri Eşleştirme ve Yükleme
    try:
        X_img, X_tab, y = loader.load_matched_data()
        
        if len(X_img) == 0:
            print("⚠️ UYARI: Klasörde eşleşen görüntü bulunamadı!")
            print("⚠️ TEST MODU: Sentetik veri ile devam ediliyor...")
            X_img = np.random.rand(50, 64, 64, 32)
            X_tab = np.random.rand(50, 7)
            y = np.random.randint(0, 2, 50)
        
        X_img = np.expand_dims(X_img, axis=-1)
        
    except Exception as e:
        print(f"Hata: {e}")
        return

    # Eğitim/Test Ayrımı
    X_i_tr, X_i_ts, X_t_tr, X_t_ts, y_tr, y_ts = train_test_split(
        X_img, X_tab, y, test_size=0.2, random_state=42
    )

    results = {}

    # --- 2. MODELLERİN EĞİTİMİ (3 MODEL) ---
    print("\n🚀 EĞİTİM SÜRECİ BAŞLIYOR (3 FARKLI MODEL)...")
    
    # 1. 3D CNN
    print("   [1/3] 3D CNN (Görüntü Modeli) Eğitiliyor...")
    cnn = build_3d_cnn(X_i_tr.shape[1:])
    cnn.fit(X_i_tr, y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    results['3D CNN'] = cnn.evaluate(X_i_ts, y_ts, verbose=0)[1]

    # 2. Random Forest
    print("   [2/3] Random Forest Eğitiliyor...")
    rf = get_ml_model()
    rf.fit(X_t_tr, y_tr)
    results['Random Forest'] = accuracy_score(y_ts, rf.predict(X_t_ts))

    # 3. Hibrit Model
    print("   [3/3] Hibrit Model (Füzyon) Test Ediliyor...")
    hybrid = HybridModel(cnn, rf)
    preds = hybrid.predict(X_i_ts, X_t_ts)
    results['Hibrit Model'] = accuracy_score(y_ts, preds)

    # --- 3. TEST ÖRNEĞİ (mpr-4) ---
    print("\n🔍 Test Dosyası (mpr-4) Analiz Ediliyor...")
    test_file = os.path.join(TEST_SAMPLE_DIR, "mpr-4.nifti.img")
    
    if not os.path.exists(test_file):
         test_file = os.path.join(TEST_SAMPLE_DIR, "mpr-4.img")

    if os.path.exists(test_file):
        vol = process_scan(test_file)
        plt.imshow(vol[:, :, 16], cmap='gray')
        plt.title("Test Hastası MR Kesiti (MPR-4)")
        plt.axis('off')
        plt.show()
        
        # Tahmin
        vol_batch = np.expand_dims(vol, axis=0)
        vol_batch = np.expand_dims(vol_batch, axis=-1)
        prob = cnn.predict(vol_batch, verbose=0)[0][0]
        
        diag = "RİSKLİ (ALZHEIMER)" if prob > 0.5 else "SAĞLIKLI"
        print(f"\n>>> TANI SONUCU: {diag} (Olasılık: {prob:.4f})")
        
        # Karşılaştırma Grafiği
        print("📊 Hasta klinik verileri genel popülasyonla kıyaslanıyor...")
        test_patient_data = {'nWBV': 0.73, 'MMSE': 26, 'eTIV': 1460}
        compare_patient_to_population(test_patient_data, df)
    else:
        print("⚠️ Test dosyası (mpr-4) bulunamadı.")

    generate_discussion(results)

if __name__ == "__main__":
    main()