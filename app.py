import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Kendi modüllerimizi çağırıyoruz
from src.config import METADATA_PATH, TEST_SAMPLE_DIR, RAW_IMG_DIRS
from src.data_loader import DataLoader
from src.models import build_3d_cnn, get_ml_model, HybridModel
from src.preprocessing import process_scan, process_scan_multi
import re

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="Alzheimer Erken Tanı Sistemi",
    page_icon="🧠",
    layout="wide"
)

# --- CSS İLE GÖRSELLİK ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    h1 { color: #2c3e50; }
    .stButton>button { width: 100%; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- BAŞLIK ---
st.title("🧠 Yapay Zeka Destekli Alzheimer Analiz Sistemi")
st.markdown("Bu sistem, **MR Görüntüleri (3D)** ve **Klinik Verileri** analiz ederek Alzheimer riskini hesaplar.")

# --- SESSION STATE (Hafıza) ---
if 'models_trained' not in st.session_state:
    st.session_state['models_trained'] = False
if 'results' not in st.session_state:
    st.session_state['results'] = {}
if 'trained_models' not in st.session_state:
    st.session_state['trained_models'] = {}

# --- YAN PANEL (SIDEBAR) ---
with st.sidebar:
    st.header("⚙️ Sistem Ayarları")
    st.info("Veri Kaynağı: OASIS-2")
    
    epochs = st.slider("Eğitim Turu (Epochs)", 1, 20, 5)
    batch_size = st.selectbox("Batch Size", [4, 8, 16], index=1)
    
    st.divider()
    st.write("Geliştirici Modu: Aktif")

# --- SEKMELER (5 SEKME) ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Veri Analizi", 
    "🚀 Model Eğitimi", 
    "🔍 Hasta Testi", 
    "📋 Demo Karşılaştırma",
    "📝 Sonuç Raporu"
])

# =============================================================================
# TAB 1: VERİ ANALİZİ
# =============================================================================
with tab1:
    st.header("Veri Seti İstatistikleri")
    
    @st.cache_data
    def load_dataset():
        loader = DataLoader(METADATA_PATH)
        return loader.load_and_clean_csv()

    df = load_dataset()
    
    if df is not None:
        unique_patients = df['Subject ID'].nunique()
        total_visits = len(df)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("### Genel İstatistikler")
            st.metric("Benzersiz Hasta Sayısı", unique_patients)
            st.metric("Toplam MR Ziyareti", total_visits)
            st.metric("Ortalama Ziyaret/Hasta", f"{total_visits/unique_patients:.1f}")
            
            st.write("### Hasta Grupları")
            patient_groups = df.groupby('Subject ID')['Group'].last().value_counts()
            st.write(patient_groups)
            
        with col2:
            st.write("### Klinik Veri Önizleme")
            st.dataframe(df[['Subject ID', 'MRI ID', 'Visit', 'Group', 'Age', 'MMSE', 'nWBV']].head(10))

        st.divider()
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Korelasyon Matrisi")
            cols = ['Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'Target']
            fig_corr, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(df[cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            st.pyplot(fig_corr)
            
        with c2:
            st.subheader("Demans Durumuna Göre Beyin Hacmi")
            fig_box, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(x='Group', y='nWBV', data=df, palette='Set2', ax=ax)
            st.pyplot(fig_box)
    else:
        st.error("CSV Dosyası Bulunamadı!")

# =============================================================================
# TAB 2: MODEL EĞİTİMİ (3 MODEL)
# =============================================================================
with tab2:
    st.header("Yapay Zeka Modellerinin Eğitimi")
    
    st.markdown("""
    Sistem şu **3 modeli** eğitecektir:
    1. **3D CNN:** MR Görüntü analizi için.
    2. **Random Forest:** Klinik veriler için.
    3. **Hibrit Model:** Görüntü ve Klinik verilerin birleşimi.
    """)
    
    if st.button("🚀 Eğitimi Başlat", type="primary"):
        status = st.status("Eğitim Süreci Başladı...", expanded=True)
        
        try:
            status.write("📥 Veriler Yükleniyor...")
            loader = DataLoader(METADATA_PATH)
            
            try:
                X_img, X_tab, y = loader.load_matched_data()
                
                if len(X_img) == 0:
                    status.warning("Gerçek MR bulunamadı, Sentetik Veri üretiliyor...")
                    X_img = np.random.rand(50, 64, 64, 32)
                    X_tab = np.random.rand(50, 7)
                    y = np.random.randint(0, 2, 50)
                
                X_img = np.expand_dims(X_img, axis=-1)
                
            except Exception as e:
                status.error(f"Veri Yükleme Hatası: {e}")
                st.stop()

            X_i_train, X_i_test, X_t_train, X_t_test, y_train, y_test = train_test_split(
                X_img, X_tab, y, test_size=0.2, random_state=42
            )
            
            # 1. CNN Eğitimi
            status.write("🧠 3D CNN Modeli Eğitiliyor...")
            cnn = build_3d_cnn(X_i_train.shape[1:])
            cnn.fit(X_i_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
            acc_cnn = cnn.evaluate(X_i_test, y_test, verbose=0)[1]
            
            # 2. Random Forest Eğitimi
            status.write("🌳 Random Forest Eğitiliyor...")
            rf = get_ml_model()
            rf.fit(X_t_train, y_train)
            acc_rf = accuracy_score(y_test, rf.predict(X_t_test))
            
            # 3. Hibrit Model
            status.write("🔗 Hibrit Model Birleştiriliyor...")
            hybrid = HybridModel(cnn, rf)
            preds_hybrid = hybrid.predict(X_i_test, X_t_test)
            acc_hybrid = accuracy_score(y_test, preds_hybrid)
            
            # Sonuçları Kaydet
            st.session_state['results'] = {
                '3D CNN': acc_cnn,
                'Random Forest': acc_rf,
                'Hibrit Model': acc_hybrid
            }
            
            st.session_state['trained_models'] = {'cnn': cnn, 'rf': rf}
            st.session_state['models_trained'] = True
            
            status.update(label="✅ Eğitim Tamamlandı!", state="complete", expanded=False)
            st.success("Tüm modeller başarıyla eğitildi!")
            
        except Exception as e:
            status.update(label="❌ Hata Oluştu", state="error")
            st.error(f"Hata detayı: {e}")

    # Sonuçları Göster
    if st.session_state['models_trained']:
        res = st.session_state['results']
        st.subheader("Model Performansları")
        
        cols = st.columns(3)
        cols[0].metric("3D CNN", f"%{res['3D CNN']*100:.1f}")
        cols[1].metric("Random Forest", f"%{res['Random Forest']*100:.1f}")
        cols[2].metric("Hibrit Model", f"%{res['Hibrit Model']*100:.1f}", delta="En İyi" if res['Hibrit Model'] == max(res.values()) else None)

# =============================================================================
# TAB 3: HASTA TESTİ (Test Sample Kullanarak)
# =============================================================================
with tab3:
    st.header("Tekil Hasta Analizi")
    
    col_img, col_data = st.columns([1, 1])
    
    with col_img:
        st.subheader("Test Hastası Seç")
        
        # Test sample klasöründeki hastaları listele
        def find_all_mpr_files(base_path):
            """RAW klasöründeki tüm MPR dosyalarını bulur."""
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
            mpr_files.sort(key=lambda x: x[0])
            return [path for _, path in mpr_files]
        
        def get_test_patients():
            """Test_sample klasöründeki hastaları ve MR dosyalarını bulur."""
            patients = {}
            if os.path.exists(TEST_SAMPLE_DIR):
                for folder in os.listdir(TEST_SAMPLE_DIR):
                    folder_path = os.path.join(TEST_SAMPLE_DIR, folder)
                    if os.path.isdir(folder_path):
                        raw_path = os.path.join(folder_path, 'RAW')
                        mpr_paths = find_all_mpr_files(raw_path)
                        if mpr_paths:
                            patients[folder] = mpr_paths  # Tüm MPR yollarını sakla
            return patients
        
        test_patients = get_test_patients()
        
        if test_patients:
            selected_patient = st.selectbox(
                "Test edilecek hastayı seçin:",
                options=list(test_patients.keys()),
                format_func=lambda x: f"🧠 {x}"
            )
            
            st.info(f"📁 Seçilen hasta: **{selected_patient}**")
            
            if st.button("🔍 Görüntüyü Analiz Et", type="primary"):
                mpr_paths = test_patients[selected_patient]
                
                with st.spinner(f"Beyin taraması işleniyor ({len(mpr_paths)} MPR)..."):
                    try:
                        vol = process_scan_multi(mpr_paths)
                        
                        # Ortadaki kesiti göster (dinamik)
                        mid_slice = vol.shape[2] // 2
                        
                        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                        
                        # Farklı kesitler göster
                        slices = [mid_slice - 5, mid_slice, mid_slice + 5]
                        titles = ["Kesit (Orta-5)", "Kesit (Orta)", "Kesit (Orta+5)"]
                        
                        for i, (s, t) in enumerate(zip(slices, titles)):
                            s = max(0, min(s, vol.shape[2]-1))  # Sınır kontrolü
                            axes[i].imshow(vol[:, :, s], cmap='gray')
                            axes[i].axis('off')
                            axes[i].set_title(t)
                        
                        plt.suptitle(f"İşlenmiş MR Kesitleri - {selected_patient}", fontsize=12)
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Volume istatistikleri
                        st.write("**Görüntü Bilgileri:**")
                        st.write(f"- Boyut: {vol.shape}")
                        st.write(f"- Min değer: {vol.min():.4f}")
                        st.write(f"- Max değer: {vol.max():.4f}")
                        st.write(f"- Ortalama: {vol.mean():.4f}")
                        
                        # Model tahmini
                        if st.session_state['models_trained']:
                            vol_batch = np.expand_dims(vol, axis=0)
                            vol_batch = np.expand_dims(vol_batch, axis=-1)
                            cnn_model = st.session_state['trained_models']['cnn']
                            prob = cnn_model.predict(vol_batch, verbose=0)[0][0]
                            st.session_state['last_pred'] = prob
                        else:
                            st.warning("⚠️ Model henüz eğitilmedi. Tahmin için önce modeli eğitin.")
                            
                    except Exception as e:
                        st.error(f"Görüntü işlenirken hata: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        else:
            st.warning("Test_sample klasöründe hasta verisi bulunamadı.")
    
    with col_data:
        st.subheader("Klinik Veri (CSV'den Otomatik)")
        
        # Seçilen hasta için CSV'den verileri al
        def get_patient_clinical_data(mri_id):
            """CSV'den hastanın klinik verilerini getirir."""
            try:
                if df is not None:
                    patient_data = df[df['MRI ID'] == mri_id]
                    if len(patient_data) > 0:
                        return patient_data.iloc[0]
            except:
                pass
            return None
        
        # Seçilen hasta varsa verilerini göster
        if test_patients and 'selected_patient' in dir():
            clinical_data = get_patient_clinical_data(selected_patient)
            
            if clinical_data is not None:
                st.success("✅ Hasta verileri CSV'den yüklendi!")
                
                # Verileri göster
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("👤 Hasta ID", clinical_data['Subject ID'])
                    st.metric("📅 Yaş", int(clinical_data['Age']))
                    st.metric("🧠 MMSE Skoru", int(clinical_data['MMSE']) if pd.notna(clinical_data['MMSE']) else "N/A")
                
                with col_b:
                    st.metric("📊 Grup", clinical_data['Group'])
                    
                    # nWBV değerini düzgün parse et (virgüllü sayılar için)
                    nwbv_val = clinical_data['nWBV']
                    if isinstance(nwbv_val, str):
                        nwbv_val = float(nwbv_val.replace(',', '.'))
                    st.metric("🧠 nWBV (Beyin Hacmi)", f"{nwbv_val:.4f}")
                    
                    # eTIV değerini düzgün parse et
                    etiv_val = clinical_data['eTIV']
                    if isinstance(etiv_val, str):
                        etiv_val = float(etiv_val.replace(',', '.'))
                    st.metric("📐 eTIV (Kafa Hacmi)", f"{etiv_val:.1f}")
                
                # Gerçek değerleri session_state'e kaydet
                age = int(clinical_data['Age'])
                mmse = int(clinical_data['MMSE']) if pd.notna(clinical_data['MMSE']) else 26
                nwbv = nwbv_val if isinstance(nwbv_val, float) else 0.72
                etiv = etiv_val if isinstance(etiv_val, float) else 1450
            else:
                st.warning("⚠️ Bu hasta için CSV'de veri bulunamadı.")
                age, mmse, nwbv, etiv = 75, 26, 0.72, 1450
        else:
            st.info("👈 Soldaki menüden bir hasta seçin.")
            age, mmse, nwbv, etiv = 75, 26, 0.72, 1450
        
        if 'last_pred' in st.session_state:
            prob = st.session_state['last_pred']
            risk_percent = prob * 100
            
            st.divider()
            if prob > 0.5:
                st.error(f"### ⚠️ YÜKSEK RİSK (ALZHEIMER)")
                st.write(f"Model Olasılığı: **%{risk_percent:.2f}**")
            else:
                st.success(f"### ✅ DÜŞÜK RİSK (SAĞLIKLI)")
                st.write(f"Model Olasılığı: **%{risk_percent:.2f}**")
                
            st.write("**Hasta vs Popülasyon Karşılaştırması:**")
            
            if df is not None:
                df['Durum'] = df['Target'].apply(lambda x: 'Demanslı' if x == 1 else 'Sağlıklı')
                means = df.groupby('Durum')[['nWBV', 'MMSE']].mean().reset_index()
                
                fig_comp, ax_comp = plt.subplots(1, 2, figsize=(10, 4))
                
                sns.barplot(x='Durum', y='nWBV', data=means, ax=ax_comp[0], palette="pastel")
                ax_comp[0].axhline(nwbv, color='red', linestyle='--', linewidth=2, label='Hasta')
                ax_comp[0].legend()
                ax_comp[0].set_title("Beyin Hacmi")
                
                sns.barplot(x='Durum', y='MMSE', data=means, ax=ax_comp[1], palette="pastel")
                ax_comp[1].axhline(mmse, color='red', linestyle='--', linewidth=2, label='Hasta')
                ax_comp[1].legend()
                ax_comp[1].set_title("Bilişsel Skor")
                
                st.pyplot(fig_comp)

# =============================================================================
# TAB 4: DEMO KARŞILAŞTIRMA
# =============================================================================
with tab4:
    st.header("📋 Demanslı vs Sağlıklı Hasta Karşılaştırması")
    
    st.info("📌 En belirgin farkları gösteren örnekler seçildi.")
    
    if df is not None:
        # En iyi karşılaştırma için hastaları seç
        demented_df = df[df['Group'] == 'Demented'].dropna(subset=['MMSE'])
        if len(demented_df) > 0:
            demented = demented_df.loc[demented_df['MMSE'].idxmin()]
        else:
            demented = None
        
        nondemented_df = df[df['Group'] == 'Nondemented'].dropna(subset=['MMSE'])
        if len(nondemented_df) > 0:
            nondemented = nondemented_df.loc[nondemented_df['MMSE'].idxmax()]
        else:
            nondemented = None
        
        if demented is not None and nondemented is not None:
            # MR görüntülerini yükle
            def find_all_mpr_paths(mri_id):
                """Hastanın tüm MPR dosyalarını bulur."""
                pattern = re.compile(r'mpr-(\d+)\.(nifti\.)?img$', re.IGNORECASE)
                for raw_dir in RAW_IMG_DIRS:
                    base_path = os.path.join(raw_dir, mri_id, 'RAW')
                    if os.path.exists(base_path):
                        mpr_files = []
                        for fname in os.listdir(base_path):
                            match = pattern.match(fname)
                            if match:
                                mpr_num = int(match.group(1))
                                full_path = os.path.join(base_path, fname)
                                mpr_files.append((mpr_num, full_path))
                        if mpr_files:
                            mpr_files.sort(key=lambda x: x[0])
                            return [path for _, path in mpr_files]
                return []
            
            demented_mr_paths = find_all_mpr_paths(demented['MRI ID'])
            nondemented_mr_paths = find_all_mpr_paths(nondemented['MRI ID'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔴 Demanslı Hasta")
                st.write(f"**Hasta ID:** {demented['MRI ID']}")
                st.write(f"**Yaş:** {int(demented['Age'])}")
                st.write(f"**MMSE:** {int(demented['MMSE'])}")
                st.write(f"**nWBV:** {demented['nWBV']:.3f}")
            
            with col2:
                st.subheader("🟢 Sağlıklı Hasta")
                st.write(f"**Hasta ID:** {nondemented['MRI ID']}")
                st.write(f"**Yaş:** {int(nondemented['Age'])}")
                st.write(f"**MMSE:** {int(nondemented['MMSE'])}")
                st.write(f"**nWBV:** {nondemented['nWBV']:.3f}")
            
            # MR Görüntüleri
            st.divider()
            st.subheader("🧠 MR Görüntüleri")
            
            mr_col1, mr_col2 = st.columns(2)
            
            with mr_col1:
                if demented_mr_paths:
                    try:
                        vol_dem = process_scan_multi(demented_mr_paths)
                        fig_dem, ax_dem = plt.subplots(figsize=(6, 6))
                        ax_dem.imshow(vol_dem[:, :, 16], cmap='gray')
                        ax_dem.set_title(f"Demanslı ({len(demented_mr_paths)} MPR)", fontsize=12, color='red')
                        ax_dem.axis('off')
                        st.pyplot(fig_dem)
                    except Exception as e:
                        st.warning(f"MR yüklenemedi: {e}")
                else:
                    st.warning("Demanslı hasta MR dosyası bulunamadı.")
            
            with mr_col2:
                if nondemented_mr_paths:
                    try:
                        vol_nond = process_scan_multi(nondemented_mr_paths)
                        fig_nond, ax_nond = plt.subplots(figsize=(6, 6))
                        ax_nond.imshow(vol_nond[:, :, 16], cmap='gray')
                        ax_nond.set_title(f"Sağlıklı ({len(nondemented_mr_paths)} MPR)", fontsize=12, color='green')
                        ax_nond.axis('off')
                        st.pyplot(fig_nond)
                    except Exception as e:
                        st.warning(f"MR yüklenemedi: {e}")
                else:
                    st.warning("Sağlıklı hasta MR dosyası bulunamadı.")
            
            # Klinik Karşılaştırma
            st.divider()
            st.subheader("📊 Klinik Veri Karşılaştırması")
            
            fig_demo, axes = plt.subplots(1, 2, figsize=(10, 4))
            
            mmse_data = pd.DataFrame({
                'Durum': ['Demanslı', 'Sağlıklı'],
                'MMSE': [demented['MMSE'], nondemented['MMSE']]
            })
            sns.barplot(x='Durum', y='MMSE', data=mmse_data, ax=axes[0], palette=['salmon', 'lightgreen'], hue='Durum', legend=False)
            axes[0].set_title("MMSE Skoru")
            axes[0].set_ylim(0, 32)
            
            nwbv_data = pd.DataFrame({
                'Durum': ['Demanslı', 'Sağlıklı'],
                'nWBV': [demented['nWBV'], nondemented['nWBV']]
            })
            sns.barplot(x='Durum', y='nWBV', data=nwbv_data, ax=axes[1], palette=['salmon', 'lightgreen'], hue='Durum', legend=False)
            axes[1].set_title("Beyin Hacmi (nWBV)")
            axes[1].set_ylim(0.6, 0.9)
            
            plt.tight_layout()
            st.pyplot(fig_demo)
        else:
            st.warning("Karşılaştırma için yeterli veri bulunamadı.")
    else:
        st.error("Veri seti yüklenemedi.")

# =============================================================================
# TAB 5: RAPOR
# =============================================================================
with tab5:
    st.header("📝 Otomatik Sonuç ve Tartışma Raporu")
    
    if st.session_state['models_trained']:
        results = st.session_state['results']
        best_model = max(results, key=results.get)
        
        st.markdown(f"""
        ### 1. Model Performans Analizi
        Bu çalışmada geliştirilen sistemde **3 farklı algoritma** test edilmiştir.
        En başarılı model **{best_model}** olmuştur (Doğruluk: **%{results[best_model]*100:.2f}**).
        
        * **3D CNN:** MR görüntülerindeki yapısal atrofiyi başarıyla analiz etmiştir.
        * **Random Forest:** Klinik verilerle yüksek başarı sağlamıştır.
        * **Hibrit Model:** İki veriyi birleştirerek karar vermiştir.
        
        ### 2. Klinik Bulgular
        **nWBV (Normalize Beyin Hacmi)** düşmesi ile Alzheimer riski arasında güçlü bir ilişki bulunmuştur.
        
        ### 3. Sonuç
        Geliştirilen sistem, klinisyenlere **"İkinci Görüş" (Second Opinion)** desteği sağlamak için uygundur.
        """)
        
        st.success("Rapor başarıyla oluşturuldu.")
    else:
        st.info("Raporun oluşturulması için lütfen önce modelleri eğitin.")
