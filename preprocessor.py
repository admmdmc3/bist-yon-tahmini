import numpy as np
from sklearn.preprocessing import MinMaxScaler

class VeriOnIsleyici:
    def __init__(self, sekans_boyutu=60):
        self.sekans_boyutu = sekans_boyutu
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def veriyi_hazirla(self, df):
        # Modelin kullanacağı özellikleri seçiyoruz
        ozellikler = ['Close', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Diff', 'SMA_20']
        veri = df[ozellikler].values
        
        # 1. KRONOLOJİK BÖLME (Data Leakage Önleme):
        # Gelecekteki fiyatların min/max değerlerinin geçmişe sızmaması için
        # veri kümesi ölçeklendirilmeden ÖNCE kronolojik olarak bölünür.
        egitim_boyutu = int(len(veri) * 0.8)
        veri_train_raw = veri[:egitim_boyutu]
        veri_test_raw = veri[egitim_boyutu:]
        
        # 2. ÖLÇEKLENDİRİCİYİ SADECE EĞİTİM SETİNE FİT ETME:
        # Test setinin istatistikleri (min, max) model eğitimine asla sızdırılmaz.
        self.scaler.fit(veri_train_raw)
        
        veri_train_scaled = self.scaler.transform(veri_train_raw)
        
        # Test setinin ilk sekansı için eğitim setinin sonundan 'sekans_boyutu' kadar geçmiş veri eklenir
        # Böylece test başlangıcındaki sekanslar geleceğe bakmadan geçmiş veriden beslenir.
        test_icin_veri = np.vstack([veri_train_raw[-self.sekans_boyutu:], veri_test_raw])
        test_icin_scaled = self.scaler.transform(test_icin_veri)
        
        # 3. ZAMAN SIRALI SEKANS OLUŞTURMA:
        X_train, y_train = [], []
        for i in range(self.sekans_boyutu, len(veri_train_scaled)):
            X_train.append(veri_train_scaled[i-self.sekans_boyutu:i]) 
            y_train.append(veri_train_scaled[i, 0]) # Hedef: Kapanış fiyatı (Close)
            
        X_test, y_test = [], []
        for i in range(self.sekans_boyutu, len(test_icin_scaled)):
            X_test.append(test_icin_scaled[i-self.sekans_boyutu:i])
            y_test.append(test_icin_scaled[i, 0])
            
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_test, y_test = np.array(X_test), np.array(y_test)
        
        print(f"Veri Hazırlandı (Data Leakage engellendi):")
        print(f"Eğitim seti boyutu: {X_train.shape}")
        print(f"Test seti boyutu: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, self.scaler
