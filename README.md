# 📈 BIST Yapay Zeka Yön Tahmin Sistemi

Bu proje, Borsa İstanbul (BIST) hisse senetlerinin günlük yön hareketlerini tahmin etmek için geliştirilmiş uçtan uca bir makine öğrenmesi web uygulamasıdır.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bist-yon-tahmini-k2gnbbdgafuchn76fas3vj.streamlit.app)

<img width="1917" height="902" alt="image" src="https://github.com/user-attachments/assets/b723ebcd-0371-437e-a493-04ae7b68e5bf" />

## 🎯 Proje Özeti
*   **Amaç:** Hisse senedi fiyat hareketlerinin yönünü (Yükseliş/Düşüş) istatistiksel metrikler ve teknik göstergelerle tahmin etmek.
*   **Makine Öğrenmesi Yaklaşımı:** Hiperparametre optimizasyonu `RandomizedSearchCV` ve zaman serilerine uygun `TimeSeriesSplit` kullanılarak gerçekleştirilmiştir.
*   **Model Başarısı:** Walk-Forward Cross Validation ile optimize edilmiş test skoru **%52.99**, tahminsel güven oranı ise **%87.58** seviyelerine ulaşmıştır.
*   **Özellik Mühendisliği (Feature Engineering):** Model karar mekanizmasında EMA50, MACD, RSI, Bollinger Bantları, Geçmiş Getiri (Lag) ve Hacim değişimleri kullanılmıştır.

## 🛠️ Kullanılan Teknolojiler
*   **Veri Bilimi ve Modelleme:** Python, Scikit-Learn, Pandas, Numpy
*   **Veri Görselleştirme:** Plotly
*   **Arayüz ve Dağıtım:** Streamlit

## 📂 Dosya Yapısı
*   `app.py`: Streamlit arayüzünü ve ana uygulama mantığını çalıştıran dosya.
*   `preprocessor.py` & `data_loader.py`: Veri çekme, temizleme ve teknik gösterge (indikatör) hesaplama modülleri.
* EDA.ipynb: Keşifçi veri analizi (EDA) ve hiperparametre optimizasyon süreçlerini içeren çalışma dosyası.

##  Kendi Bilgisayarında Çalıştırma

Projeyi klonlayın:
```bash
git clone https://github.com/admmdmc3/bist-yon-tahmini.git
cd bist-yon-tahmini

pip install -r requirements.txt
streamlit run app.py
```
