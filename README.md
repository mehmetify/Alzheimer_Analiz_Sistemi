# Alzheimer Analiz Sistemi

Bu proje, OASIS-2 MRI veri setini kullanarak Alzheimer hastalığının erken teşhisine yardımcı olan yapay zeka destekli bir karar destek sistemidir.

## Özellikler

- **3D CNN**: MR görüntülerini analiz etmek için 3 boyutlu Evrişimli Sinir Ağları kullanır.
- **Random Forest**: Klinik verileri (Yaş, Eğitim, MMSE, vb.) analiz eder.
- **Hibrit Model**: Görüntü ve klinik verilerin analiz sonuçlarını birleştirerek daha doğru tahminler yapar.
- **Web Arayüzü**: Kullanıcı dostu Streamlit arayüzü ile kolay kullanım sağlar.

## Kurulum

1. **Projeyi İndirin:**
   ```bash
   git clone <repo-url>
   cd Alzheimer_Analiz_Sistemi
   ```

2. **Gerekli Kütüphaneleri Yükleyin:**
   ```bash
   pip install -r requirements.txt
   ```

## Veri Seti Kurulumu (ÖNEMLİ)

Bu proje **OASIS-2: Longitudinal MRI Data in Nondemented and Demented Older Adults** veri setini kullanmaktadır. Veri seti boyutu nedeniyle bu depoya dahil **edilmemiştir**. Projeyi çalıştırmak için aşağıdaki adımları izleyin:

1. **Veri Setini İndirin:**
   - [OASIS Brains](https://www.oasis-brains.org/) web sitesine gidin.
   - OASIS-2 veri setini indirin.

2. **Dosyaları Yerleştirin:**
   - İndirdiğiniz veri setini projenin ana dizininde `data` adında bir klasör oluşturarak içine çıkartın.
   - Klasör yapısı şu şekilde olmalıdır:
     ```
     Alzheimer_Analiz_Sistemi/
     ├── data/
        ├── raw/
            ├── OAS2_RAW_PART1/
     ├── src/
     ├── app.py
     ...
     ```

## Kullanım

Uygulamayı başlatmak için terminalde şu komutu çalıştırın:

```bash
python -m streamlit run app.py
```

Tarayıcınızda otomatik olarak açılacaktır. Açılmazsa terminalde verilen `http://localhost:8501` adresine gidin.

## Geliştirici

- **Öğrenci No:** 23040301020
- **Ders:** Programlama Dilleri Projesi
