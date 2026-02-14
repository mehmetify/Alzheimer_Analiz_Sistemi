import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_correlation_matrix(df):
    """Korelasyon matrisi çizer."""
    cols = ['Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'Target']
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Korelasyon Matrisi")
    plt.show()

def compare_patient_to_population(patient_vals, df):
    """Hastayı genel popülasyonla karşılaştırır."""
    df['Durum'] = df['Target'].apply(lambda x: 'Demanslı' if x == 1 else 'Sağlıklı')
    means = df.groupby('Durum')[['nWBV', 'MMSE', 'eTIV']].mean().reset_index()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = ['nWBV', 'MMSE', 'eTIV']
    
    for i, metric in enumerate(metrics):
        sns.barplot(x='Durum', y=metric, data=means, ax=axes[i], palette="pastel")
        if metric in patient_vals:
            axes[i].axhline(patient_vals[metric], color='red', linestyle='--', linewidth=2, label='Hasta')
            axes[i].legend()
        axes[i].set_title(metric)
    
    plt.tight_layout()
    plt.show()

def generate_discussion(results):
    """Sonuç raporu oluşturur."""
    best = max(results, key=results.get)
    print("\n" + "="*60)
    print("       SONUÇ RAPORU       ")
    print("="*60)
    print(f"""
    MODEL SONUÇLARI:
    En Başarılı: {best} (Doğruluk: %{results[best]*100:.2f})
    
    - 3D CNN: MR görüntülerindeki atrofiyi tespit etti.
    - Random Forest: Klinik verilerle yüksek başarı sağladı.
    - Hibrit Model: İki veriyi birleştirerek karar verdi.

    KLİNİK ÇIKARIM:
    Beyin hacmi (nWBV) düştükçe Alzheimer riski artmaktadır.
    MMSE testi erken tanıda kritik öneme sahiptir.
    """)
    print("="*60)