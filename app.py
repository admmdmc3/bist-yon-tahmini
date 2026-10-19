import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components

# 1. BIST Hisse Sözlüğü (Yalnızca Yahoo Finance üzerinde aktif ve doğrulanmış BIST hisseleri)
BIST_HISSELERI = {
    "AEFES.IS": "Anadolu Efes",
    "AGHOL.IS": "Anadolu Grubu Holding",
    "AHGAZ.IS": "Ahlatcı Doğalgaz",
    "AKBNK.IS": "Akbank",
    "AKCNS.IS": "Akçansa",
    "AKFGY.IS": "Akfen GYO",
    "AKFYE.IS": "Akfen Yenilenebilir Enerji",
    "AKSA.IS": "Aksa",
    "AKSEN.IS": "Aksa Enerji",
    "ALARK.IS": "Alarko Holding",
    "ALBRK.IS": "Albaraka Türk",
    "ALFAS.IS": "Alfa Solar Enerji",
    "ARCLK.IS": "Arçelik",
    "ASELS.IS": "Aselsan",
    "ASTOR.IS": "Astor Enerji",
    "ASUZU.IS": "Anadolu Isuzu",
    "AYDEM.IS": "Aydem Enerji",
    "BAGFS.IS": "Bagfaş",
    "BANVT.IS": "Banvit",
    "BERA.IS": "Bera Holding",
    "BIMAS.IS": "BİM Mağazalar",
    "BOBET.IS": "Boğaziçi Beton",
    "BRYAT.IS": "Borusan Yatırım",
    "BUCIM.IS": "Bursa Çimento",
    "CANTE.IS": "Çan2 Termik",
    "CCOLA.IS": "Coca Cola İçecek",
    "CEMTS.IS": "Çemtaş",
    "CIMSA.IS": "Çimsa",
    "CWENE.IS": "CW Enerji",
    "DOAS.IS": "Doğuş Otomotiv",
    "DOHOL.IS": "Doğan Holding",
    "ECZYT.IS": "Eczacıbaşı Yatırım",
    "EGEEN.IS": "Ege Endüstri",
    "EKGYO.IS": "Emlak Konut GYO",
    "ENJSA.IS": "Enerjisa Enerji",
    "ENKAI.IS": "Enka İnşaat",
    "EREGL.IS": "Erdemir",
    "EUPWR.IS": "Europower Enerji",
    "FROTO.IS": "Ford Otosan",
    "GARAN.IS": "Garanti BBVA",
    "GESAN.IS": "Girişim Elektrik",
    "GLYHO.IS": "Global Yatırım Holding",
    "GUBRF.IS": "Gübre Fabrikaları",
    "GWIND.IS": "Galata Wind Enerji",
    "HALKB.IS": "Halkbank",
    "HEKTS.IS": "Hektaş",
    "ISCTR.IS": "İş Bankası (C)",
    "ISDMR.IS": "İskenderun Demir Çelik",
    "ISGYO.IS": "İş GYO",
    "KCAER.IS": "Kocaer Çelik",
    "KCHOL.IS": "Koç Holding",
    "KLSER.IS": "Kaleseramik",
    "KMPUR.IS": "Kimteks Poliüretan",
    "KONTR.IS": "Kontrolmatik",
    "KONYA.IS": "Konya Çimento",
    "KORDS.IS": "Kordsa",
    "KRDMD.IS": "Kardemir (D)",
    "KZBGY.IS": "Kızılbük GYO",
    "MAVI.IS": "Mavi Giyim",
    "MGROS.IS": "Migros",
    "ODAS.IS": "Odaş Elektrik",
    "OTKAR.IS": "Otokar",
    "OYAKC.IS": "Oyak Çimento",
    "PETKM.IS": "Petkim",
    "PGSUS.IS": "Pegasus",
    "QUAGR.IS": "Qua Granite",
    "REEDR.IS": "Reeder Teknoloji",
    "SAHOL.IS": "Sabancı Holding",
    "SASA.IS": "Sasa Polyester",
    "SDTTR.IS": "SDT Uzay ve Savunma",
    "SISE.IS": "Şişecam",
    "SKBNK.IS": "Şekerbank",
    "SMRTG.IS": "Smart Güneş Enerjisi",
    "SOKM.IS": "Şok Marketler",
    "TAVHL.IS": "TAV Havalimanları",
    "TCELL.IS": "Turkcell",
    "THYAO.IS": "Türk Hava Yolları",
    "TOASO.IS": "Tofaş",
    "TSKB.IS": "TSKB",
    "TTKOM.IS": "Türk Telekom",
    "TTRAK.IS": "Türk Traktör",
    "TUPRS.IS": "Tüpraş",
    "ULKER.IS": "Ülker",
    "VAKBN.IS": "VakıfBank",
    "VESBE.IS": "Vestel Beyaz Eşya",
    "VESTL.IS": "Vestel",
    "YEOTK.IS": "YEO Teknoloji",
    "YKBNK.IS": "Yapı Kredi",
    "YYLGD.IS": "Yayla Agro Gıda",
    "ZOREN.IS": "Zorlu Enerji"
}

# BIST Sinyal Tarayıcı (Batch Screener) için varsayılan en likit 10 lokomotif hisse
VARSAYILAN_TARAMA_HISSELERI = [
    "THYAO.IS",
    "GARAN.IS",
    "ASELS.IS",
    "EREGL.IS",
    "KCHOL.IS",
    "AKBNK.IS",
    "TUPRS.IS",
    "SISE.IS",
    "BIMAS.IS",
    "YKBNK.IS"
]

# Görünüm Periyodu Haritası (Sadece grafiklerin filtrelenmesi/yakınlaştırılması için kullanılır)
GORUNUM_GUN_HARITASI = {
    "1 Ay": 30,
    "3 Ay": 90,
    "6 Ay": 180,
    "1 Yıl": 365,
    "3 Yıl": 3 * 365
}

# Dinamik Tahmin Ufku (Prediction Horizon) Haritası
TAHMIN_UFKU_HARITASI = {
    "1 Gün (Yarınki Yön)": 1,
    "3 İş Günü (Kısa Vade Trend)": 3,
    "5 İş Günü (Haftalık Trend)": 5
}

TAHMIN_UFKU_BASLIKLARI = {
    1: "Yarınki Yön Tahmini",
    3: "3 İş Günü Sonrası Trend Tahmini",
    5: "5 İş Günü Sonrası Trend Tahmini"
}

# Haber Akışı için Türkçe Ay İsimleri Sözlüğü
TURKCE_AYLAR = {
    1: "Ocak",
    2: "Şubat",
    3: "Mart",
    4: "Nisan",
    5: "Mayıs",
    6: "Haziran",
    7: "Temmuz",
    8: "Ağustos",
    9: "Eylül",
    10: "Ekim",
    11: "Kasım",
    12: "Aralık"
}

# Sayfa Durum Yönetimi
if "analiz_aktif" not in st.session_state:
    st.session_state.analiz_aktif = False
if "screener_sonuc_df" not in st.session_state:
    st.session_state.screener_sonuc_df = None
if "screener_ufuk" not in st.session_state:
    st.session_state.screener_ufuk = 1

st.set_page_config(
    page_title="BIST Yön Tahmin Paneli",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Mobil ve Masaüstü Ekranlarda Metrik Kartı Yazı Taşmalarını Önleyen Responsive CSS
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] {
        font-size: clamp(1.25rem, 2.2vw, 1.85rem) !important;
        white-space: normal !important;
        overflow-wrap: break-word !important;
        line-height: 1.25 !important;
    }
    [data-testid="stMetricDelta"] > div {
        white-space: normal !important;
        overflow-wrap: break-word !important;
        font-size: clamp(0.78rem, 1.3vw, 0.92rem) !important;
        line-height: 1.2 !important;
    }
    [data-testid="stMetricLabel"] > div {
        white-space: normal !important;
        overflow-wrap: break-word !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Statik grafikler için araç çubuğunu gizleyen Plotly yapılandırması
PLOTLY_STATIK_CONFIG = {
    "displayModeBar": False,
    "displaylogo": False
}

# Zaman serisi grafikleri için güvenli kaydırma/sıfırlama yapılandırması (Sınırlandırılmış scroll zoom aktif)
PLOTLY_ZAMAN_SERISI_CONFIG = {
    "displayModeBar": False,
    "scrollZoom": True,
    "doubleClick": "reset"
}

# Modelin kullandığı güçlendirilmiş öznitelikler (Feature Engineering)
OZELLIKLER = [
    'RSI',
    'MACD',
    'MACD_Signal',
    'BB_PBand',
    'Volume_Change',
    'Return_Lag1',
    'Return_Lag2',
    'RSI_Diff3',
    'EMA20_Dist',
    'EMA50_Dist',
    'ATR_Pct',
    'Volatility_10D'
]

OZELLIK_ETIKETLERI = {
    'RSI': 'RSI (14)',
    'MACD': 'MACD',
    'MACD_Signal': 'MACD Sinyal',
    'BB_PBand': 'Bollinger %B',
    'Volume_Change': 'Hacim Değişimi',
    'Return_Lag1': 'Getiri (Lag-1)',
    'Return_Lag2': 'Getiri (Lag-2)',
    'RSI_Diff3': 'RSI 3G Eğim (Diff3)',
    'EMA20_Dist': 'EMA20 Uzaklık (%)',
    'EMA50_Dist': 'EMA50 Uzaklık (%)',
    'ATR_Pct': 'ATR (14) Yüzde',
    'Volatility_10D': '10G Volatilite (Std)'
}

# --- FONKSİYONLAR ---

@st.cache_data(ttl=3600)
def veri_hazirla(sembol, horizon=1):
    # EĞİTİM VERİSİ PERİYODU İLE GÖRÜNÜM PERİYODUNUN AYRILMASI:
    # Kullanıcının seçtiği grafik görünüm periyodundan bağımsız olarak, istatistiksel çöküşü
    # ve aşırı öğrenmeyi (overfitting) önlemek adına model her zaman 3 yıllık geçmiş veri
    # (+ 60 günlük indikatör ısınma tamponu) üzerinden eğitilir.
    egitim_gun_sayisi = 3 * 365
    tampon_gun = 60
    bitis = datetime.now()
    baslangic = bitis - timedelta(days=egitim_gun_sayisi + tampon_gun)
    df = yf.download(sembol, start=baslangic, end=bitis, progress=False)

    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 1. SIZINTISIZ ÖZNİTELİK MÜHENDİSLİĞİ (Feature Engineering - Zero Data Leakage):
    # Tüm göstergeler yalnızca t anı ve geçmiş fiyatlarla hesaplanır (Lookahead Bias yoktur).
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    macd_obj = MACD(close=df['Close'])
    df['MACD'] = macd_obj.macd()
    df['MACD_Signal'] = macd_obj.macd_signal()

    # Bollinger Bantları (Üst, Alt ve %B konumu) + Hacim değişim oranı
    bb_obj = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_High'] = bb_obj.bollinger_hband()
    df['BB_Low'] = bb_obj.bollinger_lband()
    df['BB_PBand'] = bb_obj.bollinger_pband()
    df['Volume_Change'] = df['Volume'].pct_change()

    # Gecikmeli Değişkenler (Lag Features): Dünün ve 2 gün öncesinin getirisi + RSI 3 günlük farkı
    gunluk_getiri = df['Close'].pct_change()
    df['Return_Lag1'] = gunluk_getiri.shift(1)
    df['Return_Lag2'] = gunluk_getiri.shift(2)
    df['RSI_Diff3'] = df['RSI'].diff(3)

    # Trend Uzaklığı: EMA20 ve EMA50 serileri ve fiyatın bu ortalamalara yüzdesel uzaklığı
    df['EMA20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
    df['EMA50'] = EMAIndicator(close=df['Close'], window=50).ema_indicator()
    df['EMA20_Dist'] = (df['Close'] - df['EMA20']) / df['EMA20']
    df['EMA50_Dist'] = (df['Close'] - df['EMA50']) / df['EMA50']

    # Volatilite: Normalize edilmiş 14 günlük ATR ve 10 günlük getiri standart sapması
    atr14 = AverageTrueRange(
        high=df['High'], low=df['Low'], close=df['Close'], window=14
    ).average_true_range()
    df['ATR_Pct'] = atr14 / df['Close']
    df['Volatility_10D'] = gunluk_getiri.rolling(window=10).std()

    # 2. DİNAMİK HEDEF DEĞİŞKEN (Target) VE SHIFT KONTROLÜ:
    # Özellikler (X) bugünün verisidir (t).
    # Hedef (y), seçilen tahmin ufku (horizon) sonrasındaki kapanışın bugünden yüksek olup olmadığıdır:
    # (Close_{t + horizon} > Close_t). shift(-horizon) sadece hedef değişkende kullanılır.
    df['Target'] = (df['Close'].shift(-horizon) > df['Close']).astype(int)

    # Isınma tamponundaki ilk 60 günü temizle (EMA50, MACD, RSI ve ATR tamamen oturmuştur)
    df = df.iloc[tampon_gun:].copy()

    # NaN ve Sonsuz (inf) Değer Temizliği:
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def model_egit_ve_tahmin(df, horizon=1):
    # SIZINTI ENGELLEME (Dinamik Tahmin Ufku Kontrolü):
    # Son 'horizon' kadar günün gerçek hedefi (t + horizon kapanışı) henüz gerçekleşmemiştir.
    # Bu nedenle model eğitimi ve değerlendirmesi sırasında son 'horizon' gün eğitimden hariç tutulur;
    # ancak bugünkü canlı tahmin (iloc[-1]) için en son günün öznitelikleri kullanılır.
    df_egitim = df.iloc[:-horizon].copy()
    son_gun_verisi = df.iloc[[-1]][OZELLIKLER]
    dunku_gun_verisi = df.iloc[[-2]][OZELLIKLER] if len(df) >= 2 else son_gun_verisi

    X = df_egitim[OZELLIKLER]
    y = df_egitim['Target']

    # Kronolojik olarak ilk %80 Eğitim + Walk-Forward CV, son %20 Out-of-Sample Test için ayrılır
    split_noktasi = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_noktasi], X.iloc[split_noktasi:]
    y_train, y_test = y.iloc[:split_noktasi], y.iloc[split_noktasi:]

    # Sınıf dengesizliğini ve "Bullish Bias" (sürekli yükseliş tahmin etme) sorununu
    # dengelemek için eğitim verisindeki sınıf oranlarına göre dinamik scale_pos_weight hesabı
    negatif_sayisi = int((y_train == 0).sum())
    pozitif_sayisi = int((y_train == 1).sum())
    scale_pos_weight = negatif_sayisi / pozitif_sayisi if pozitif_sayisi > 0 else 1.0

    # SKLEARN PIPELINE MİMARİSİ:
    # StandardScaler'ı Pipeline içine alarak TimeSeriesSplit'in her bir iç katlamasında (fold)
    # yalnız o katlamanın eğitim verisine fit edilmesini garanti ediyoruz (%0 Veri Sızıntısı).
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight
        ))
    ])

    param_grid = {
        'xgb__n_estimators': [50, 100],
        'xgb__max_depth': [3, 4],
        'xgb__learning_rate': [0.05, 0.1]
    }

    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=tscv,
        scoring='accuracy'
    )
    grid_search.fit(X_train, y_train)

    # Walk-Forward CV başarısı ve en iyi Pipeline modeli doğrudan GridSearchCV'den alınır
    cv_basari = float(grid_search.best_score_)
    final_model = grid_search.best_estimator_

    # Bugünün (iloc[-1]) ve dünün (iloc[-2]) tahmin ile olasılık değerleri
    tahmin = int(final_model.predict(son_gun_verisi)[0])
    olasilik = final_model.predict_proba(son_gun_verisi)[0]

    dunku_tahmin = int(final_model.predict(dunku_gun_verisi)[0])
    dunku_olasilik = final_model.predict_proba(dunku_gun_verisi)[0]
    dunku_guven = float(dunku_olasilik[1] if dunku_tahmin == 1 else dunku_olasilik[0])

    # Out-of-Sample test seti tahminleri ve olasılıkları (CSV Rapor Dışa Aktarma için)
    y_pred = final_model.predict(X_test)
    y_test_proba = final_model.predict_proba(X_test)
    y_test_guven_serisi = pd.Series(
        np.where(y_pred == 1, y_test_proba[:, 1], y_test_proba[:, 0]) * 100.0,
        index=y_test.index
    )

    return (
        tahmin,
        olasilik,
        dunku_tahmin,
        dunku_guven,
        cv_basari,
        final_model,
        y_test,
        y_pred,
        y_test_guven_serisi
    )


def hizli_sinyal_tara(sembol, horizon=1):
    """Toplu BIST Sinyal Tarayıcı (Batch Screener) için GridSearchCV yükü olmadan
    TimeSeriesSplit(n_splits=3) ve Pipeline kullanan sızıntısız hızlı modelleme fonksiyonu."""
    df = veri_hazirla(sembol, horizon=horizon)
    if df is None or len(df) <= (horizon + 40):
        return None

    df_egitim = df.iloc[:-horizon].copy()
    son_gun_verisi = df.iloc[[-1]][OZELLIKLER]

    X = df_egitim[OZELLIKLER]
    y = df_egitim['Target']

    negatif_sayisi = int((y == 0).sum())
    pozitif_sayisi = int((y == 1).sum())
    scale_pos_weight = negatif_sayisi / pozitif_sayisi if pozitif_sayisi > 0 else 1.0

    batch_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.08,
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight
        ))
    ])

    tscv = TimeSeriesSplit(n_splits=3)
    cv_skorlari = cross_val_score(batch_pipeline, X, y, cv=tscv, scoring='accuracy')
    cv_dogruluk = float(np.mean(cv_skorlari)) * 100.0

    batch_pipeline.fit(X, y)
    tahmin = int(batch_pipeline.predict(son_gun_verisi)[0])
    olasilik = batch_pipeline.predict_proba(son_gun_verisi)[0]
    guven_yuzde = float(olasilik[1] if tahmin == 1 else olasilik[0]) * 100.0

    son_fiyat = float(df['Close'].iloc[-1])
    guncel_rsi = float(df['RSI'].iloc[-1])

    return {
        "Hisse Kodu": sembol,
        "Şirket Adı": BIST_HISSELERI.get(sembol, sembol.replace(".IS", "")),
        "Son Fiyat (TL)": round(son_fiyat, 2),
        "Sinyal": "🟢 YÜKSELİŞ" if tahmin == 1 else "🔴 DÜŞÜŞ",
        "Tahmin Güven Oranı (%)": round(guven_yuzde, 2),
        "Walk-Forward CV Doğruluğu (%)": round(cv_dogruluk, 2),
        "Güncel RSI (14)": round(guncel_rsi, 1)
    }


def tradingview_grafik_ciz(df, sembol):
    # 2 Satırlı (2-Row) Plotly Subplots Mimarisi:
    # Üst Panel (Row 1 - %70): Mum Grafiği + EMA 20 + EMA 50 + Bollinger Bantları (Üst/Alt)
    # Alt Panel (Row 2 - %30): RSI (14) + 30/70 Aşırı Satım / Aşırı Alım Referans Çizgileri
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.7, 0.3],
        subplot_titles=(
            f"{sembol} Fiyat Grafiği, EMA (20/50) & Bollinger Bantları",
            "RSI (14) Momentum Göstergesi"
        )
    )

    # 1. ÜST PANEL: Mum Grafiği (Candlestick)
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Fiyat (OHLC)",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a",
            decreasing_fillcolor="#ef5350"
        ),
        row=1,
        col=1
    )

    # ÜST PANEL: EMA 20 ve EMA 50
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["EMA20"],
            mode="lines",
            name="EMA 20",
            line=dict(color="#ff9800", width=1.6)
        ),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["EMA50"],
            mode="lines",
            name="EMA 50",
            line=dict(color="#2962ff", width=1.8)
        ),
        row=1,
        col=1
    )

    # ÜST PANEL: Bollinger Bantları (Üst ve Alt Bant - yarı saydam dolgu ile)
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["BB_High"],
            mode="lines",
            name="Bollinger Üst Bant",
            line=dict(color="rgba(179, 157, 219, 0.55)", width=1.2, dash="dot")
        ),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["BB_Low"],
            mode="lines",
            name="Bollinger Alt Bant",
            line=dict(color="rgba(179, 157, 219, 0.55)", width=1.2, dash="dot"),
            fill="tonexty",
            fillcolor="rgba(179, 157, 219, 0.08)"
        ),
        row=1,
        col=1
    )

    # 2. ALT PANEL: RSI (14) Çizgisi ve 30/70 Referans Seviyeleri
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["RSI"],
            mode="lines",
            name="RSI (14)",
            line=dict(color="#ab47bc", width=2)
        ),
        row=2,
        col=1
    )

    # 70 Aşırı Alım (Kırmızı kesikli çizgi) ve 30 Aşırı Satım (Yeşil kesikli çizgi)
    fig.add_hline(
        y=70,
        line_dash="dash",
        line_color="#ef5350",
        line_width=1.3,
        annotation_text="Aşırı Alım (70)",
        annotation_position="top right",
        annotation_font_color="#ef5350",
        row=2,
        col=1
    )
    fig.add_hline(
        y=30,
        line_dash="dash",
        line_color="#26a69a",
        line_width=1.3,
        annotation_text="Aşırı Satım (30)",
        annotation_position="bottom right",
        annotation_font_color="#26a69a",
        row=2,
        col=1
    )

    # Y ekseni fiyat sınırları (%5 alt/üst pay ile sabitlenmiş) ve X ekseni 15 günlük sağ analist boşluğu
    fiyat_alt_sinir = float(df[["Low", "BB_Low", "EMA20", "EMA50"]].min().min()) * 0.95
    fiyat_ust_sinir = float(df[["High", "BB_High", "EMA20", "EMA50"]].max().max()) * 1.05
    x_baslangic = df.index[0]
    x_bitis_bosluklu = df.index[-1] + pd.Timedelta(days=15)

    fig.update_layout(
        template="plotly_dark",
        height=660,
        dragmode="pan",
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="left",
            x=0
        )
    )

    fig.update_xaxes(
        nticks=8,
        tickformat="%d %b '%y",
        autorange=False,
        range=[x_baslangic, x_bitis_bosluklu],
        minallowed=x_baslangic,
        maxallowed=x_bitis_bosluklu,
        fixedrange=False,
        row=1,
        col=1
    )
    fig.update_xaxes(
        nticks=8,
        tickformat="%d %b '%y",
        autorange=False,
        range=[x_baslangic, x_bitis_bosluklu],
        minallowed=x_baslangic,
        maxallowed=x_bitis_bosluklu,
        fixedrange=False,
        title_text="Tarih",
        row=2,
        col=1
    )
    fig.update_yaxes(
        title_text="Fiyat (TL)",
        autorange=False,
        range=[fiyat_alt_sinir, fiyat_ust_sinir],
        minallowed=fiyat_alt_sinir,
        maxallowed=fiyat_ust_sinir,
        fixedrange=True,
        row=1,
        col=1
    )
    fig.update_yaxes(
        title_text="RSI",
        autorange=False,
        range=[0, 100],
        minallowed=0,
        maxallowed=100,
        fixedrange=True,
        row=2,
        col=1
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False, "scrollZoom": True, "doubleClick": "reset"}
    )


def lokal_xai_aciklama_uret(df, sembol, tahmin, guven, model, ufuk_basligi):
    """Son günkü (iloc[-1]) gerçek indikatör değerlerini ve modelin en önemli özniteliklerini
    okuyarak dinamik Türkçe karar gerekçesi (Lokal XAI) metni üretir."""
    son_satir = df.iloc[-1]

    rsi_degeri = float(son_satir["RSI"])
    if rsi_degeri >= 70:
        rsi_durum = "Aşırı Alım"
    elif rsi_degeri <= 30:
        rsi_durum = "Aşırı Satım"
    else:
        rsi_durum = "Nötr"

    bb_degeri = float(son_satir["BB_PBand"])
    if bb_degeri >= 0.80:
        bb_yorum = "üst banda yakın"
    elif bb_degeri <= 0.20:
        bb_yorum = "alt banda yakın"
    else:
        bb_yorum = "bant içi dengeli bölgede"

    ema20_fark = float(son_satir["EMA20_Dist"]) * 100
    ema_konum = "üzerinde" if ema20_fark >= 0 else "altında"

    hacim_degisim = float(son_satir["Volume_Change"]) * 100
    yon_metni = "YÜKSELİŞ" if tahmin == 1 else "DÜŞÜŞ"

    # Modelin en yüksek ağırlığa sahip 3 özniteliğini belirle
    xgb_step = model.named_steps["xgb"]
    onem_sirasi = np.argsort(xgb_step.feature_importances_)[::-1][:3]
    en_onemli_3 = ", ".join([OZELLIK_ETIKETLERI[OZELLIKLER[i]] for i in onem_sirasi])

    aciklama = (
        f"🧠 **Modelin Bugünkü Karar Gerekçesi (Lokal Analiz - {ufuk_basligi}):** "
        f"Model bugün **{sembol}** için **%{guven * 100:.2f} güvenle {yon_metni}** öngörüyor. "
        f"Modelin karar mekanizmasında en belirleyici 3 faktör **{en_onemli_3}** oldu. "
        f"Kararda etkili olan güncel teknik göstergeler: RSI değeri **{rsi_degeri:.1f}** seviyesinde (**{rsi_durum}**), "
        f"Bollinger %B **{bb_degeri:.2f}** noktasında ({bb_yorum}), "
        f"fiyat 20 günlük ortalamasının **%{abs(ema20_fark):.2f}** **{ema_konum}** ve "
        f"günlük hacim değişimi **%{hacim_degisim:+.1f}**."
    )
    return aciklama


def rss_tarih_parse_ve_formatla(pub_date_str):
    """RSS pubDate tarihini parse ederek sıralama için datetime nesnesi ve
    ekranda gösterim için Türkçe okunaklı tarih metni ('12 Ağustos 2026, 10:00') döndürür."""
    if not pub_date_str:
        return datetime.min.replace(tzinfo=timezone.utc), "Tarih Belirtilmemiş"
    try:
        dt = parsedate_to_datetime(pub_date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        # Türkiye saat dilimine (UTC+3) çevir
        tr_tz = timezone(timedelta(hours=3))
        dt_tr = dt.astimezone(tr_tz)
        ay_adi = TURKCE_AYLAR.get(dt_tr.month, "")
        formatli_tarih = f"{dt_tr.day} {ay_adi} {dt_tr.year}, {dt_tr.strftime('%H:%M')}"
        return dt_tr, formatli_tarih
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc), pub_date_str


# --- STREAMLIT ARAYÜZÜ ---

st.title("📈 BIST Yapay Zeka Yön Tahmin Sistemi")

# Yan Menü (Sidebar) - Form Kilidi: "Analizi Başlat" tuşuna basılmadan hiçbir parametre devreye girmez
with st.sidebar.form("analiz_parametre_formu"):
    st.header("Hisse & Parametre Seçimi")

    # Selectbox Hazırlığı (Hisse Kodu - Hisse Adı)
    secenekler = [f"{kod} - {ad}" for kod, ad in BIST_HISSELERI.items()]
    secilen_hisse_str = st.selectbox("Tahmin Edilecek Hisseyi Seçin:", secenekler)

    # String Parçalama: Sol taraftaki kodu al (Örn: 'THYAO.IS')
    secilen_sembol = secilen_hisse_str.split(" - ")[0]

    # Dinamik Tahmin Ufku (Prediction Horizon) Seçimi
    secilen_ufuk_str = st.selectbox(
        "Tahmin Ufku (Prediction Horizon):",
        list(TAHMIN_UFKU_HARITASI.keys()),
        index=0,
        help="Modelin kaç iş günü sonrasının kapanış yönünü tahmin edeceğini belirler."
    )

    # Grafik Görünüm Aralığı Filtresi
    secilen_periyot = st.selectbox(
        "Grafik Görünüm Periyodu:",
        ["1 Ay", "3 Ay", "6 Ay", "1 Yıl", "3 Yıl"],
        index=3
    )
    st.caption(
        "Not: İstatistiksel güvenilirliği korumak ve aşırı öğrenmeyi (overfitting) önlemek "
        "için yapay zeka modeli arka planda her zaman **3 yıllık** geçmiş veriyle eğitilir. "
        "Buradaki seçim yalnızca grafiklerin gösterim/yakınlaştırma aralığını filtreler."
    )

    st.markdown("---")
    st.subheader("💰 Backtest Simülatör Ayarları")
    baslangic_sermayesi = st.number_input(
        "Başlangıç Sermayesi (TL):",
        min_value=1000,
        max_value=100000000,
        value=10000,
        step=1000
    )
    komisyon_orani_yuzde = st.slider(
        "İşlem Komisyon Oranı (%):",
        min_value=0.0,
        max_value=1.0,
        value=0.15,
        step=0.01,
        help="Yalnızca pozisyon değişimlerinde (nakitten hisseye geçiş veya hisseden nakite çıkış) kesilen tek yönlü aracı kurum komisyon oranı."
    )

    tahmin_butonu = st.form_submit_button("Analizi Başlat", type="primary", use_container_width=True)

if tahmin_butonu:
    st.session_state.analiz_aktif = True
    st.session_state.sidebar_kapat_tetikle = True
    st.session_state.aktif_parametreler = {
        "sembol": secilen_sembol,
        "hisse_str": secilen_hisse_str,
        "ufuk_str": secilen_ufuk_str,
        "horizon": TAHMIN_UFKU_HARITASI[secilen_ufuk_str],
        "periyot": secilen_periyot,
        "sermaye": baslangic_sermayesi,
        "komisyon": komisyon_orani_yuzde
    }

if st.session_state.get("sidebar_kapat_tetikle", False):
    components.html(
        """
        <script>
        const parentDoc = window.parent.document;
        const collapseBtn = parentDoc.querySelector('[data-testid="stSidebarCollapseButton"] button')
                         || parentDoc.querySelector('[data-testid="collapsedControl"] button')
                         || parentDoc.querySelector('section[data-testid="stSidebar"] button[kind="header"]')
                         || parentDoc.querySelector('section[data-testid="stSidebar"] button');
        if (collapseBtn) {
            collapseBtn.click();
        }
        </script>
        """,
        height=0,
        width=0
    )
    st.session_state.sidebar_kapat_tetikle = False

if st.session_state.analiz_aktif and "aktif_parametreler" in st.session_state:
    aktif = st.session_state.aktif_parametreler
    secilen_sembol = aktif["sembol"]
    secilen_hisse_str = aktif["hisse_str"]
    secilen_ufuk_str = aktif["ufuk_str"]
    horizon = aktif["horizon"]
    ufuk_basligi = TAHMIN_UFKU_BASLIKLARI.get(horizon, "Yarınki Yön Tahmini")
    secilen_periyot = aktif["periyot"]
    baslangic_sermayesi = aktif["sermaye"]
    komisyon_orani_yuzde = aktif["komisyon"]

    with st.spinner(f'{secilen_sembol} verileri ({secilen_ufuk_str}) analiz ediliyor...'):
        data = veri_hazirla(secilen_sembol, horizon=horizon)

        if data is not None and len(data) > (horizon + 30):
            (
                tahmin,
                olasilik,
                dunku_tahmin,
                dunku_guven,
                basari,
                model,
                y_test,
                y_pred,
                y_test_guven_serisi
            ) = model_egit_ve_tahmin(data, horizon=horizon)

            guncel_fiyat = data['Close'].iloc[-1]

            # Seçilen görünüm periyoduna göre grafik verisini filtrele
            gorunum_gun = GORUNUM_GUN_HARITASI.get(secilen_periyot, 365)
            gorunum_baslangic = data.index[-1] - timedelta(days=gorunum_gun)
            df_gorunum = data[data.index >= gorunum_baslangic].copy()
            if df_gorunum.empty:
                df_gorunum = data.copy()

            # Günlük Fiyat Değişimini (Delta) Hesaplama
            if len(data) > 1:
                onceki_kapanis = data['Close'].iloc[-2]
                fiyat_degisim_yuzde = ((guncel_fiyat - onceki_kapanis) / onceki_kapanis) * 100
                fiyat_delta = f"{fiyat_degisim_yuzde:+.2f}%"
            else:
                fiyat_delta = "+0.00%"

            # --- SEKMELİ ANA YERLEŞİM (5 SEKME) ---
            sekme_fiyat, sekme_model, sekme_istatistik, sekme_tarayici, sekme_haber = st.tabs([
                "📈 Fiyat ve Tahmin",
                "🧠 Model Analizi",
                "📊 İstatistiksel Metrikler",
                "🔍 BIST Sinyal Tarayıcı (Screener)",
                "📰 Haber Akışı"
            ])

            # 1. SEKME: Fiyat metrikleri + tahmin sonucu + Lokal XAI + 2 satırlı mum/indikatör grafiği
            with sekme_fiyat:
                col1, col2, col3 = st.columns(3)
                col1.metric("Güncel Fiyat", f"{guncel_fiyat:.2f} TL", delta=fiyat_delta)

                cv_delta = f"{(basari - 0.50)*100:+.2f}% (vs %50 Baz)"
                col2.metric(
                    "Model Doğruluğu (Walk-Forward CV)",
                    f"%{basari*100:.2f}",
                    delta=cv_delta
                )

                guven = float(olasilik[1] if tahmin == 1 else olasilik[0])
                if tahmin == dunku_tahmin:
                    guven_delta = f"{(guven - dunku_guven)*100:+.2f}% (Düne Göre)"
                    guven_delta_renk = "normal"
                else:
                    guven_delta = "Sinyal Yön Değişti 🔄"
                    guven_delta_renk = "off"

                col3.metric(
                    "Tahmin Güven Oranı",
                    f"%{guven*100:.2f}",
                    delta=guven_delta,
                    delta_color=guven_delta_renk
                )

                st.markdown("---")
                if tahmin == 1:
                    st.success(f"### 🚀 {secilen_sembol} {ufuk_basligi}: **YÜKSELİŞ**")
                else:
                    st.error(f"### 📉 {secilen_sembol} {ufuk_basligi}: **DÜŞÜŞ**")

                # Lokal Açıklanabilir Yapay Zeka (XAI) Karar Gerekçesi Kutusu
                xai_metni = lokal_xai_aciklama_uret(
                    data, secilen_sembol, tahmin, guven, model, ufuk_basligi
                )
                st.info(xai_metni)

                # 2 Satırlı Mum + EMA20/50 + Bollinger + RSI(14) Grafiği
                tradingview_grafik_ciz(df_gorunum, secilen_sembol)

            # 2. SEKME: Feature Importance + Out-of-Sample Doğrulama Tablosu + Pozisyon Korumalı Backtest + CSV Export
            with sekme_model:
                sub_col1, sub_col2 = st.columns(2)

                with sub_col1:
                    st.subheader("🔍 Model Özellik Önemi (Feature Importance)")
                    xgb_step = model.named_steps['xgb']
                    importance_df = pd.DataFrame({
                        "Özellik": [OZELLIK_ETIKETLERI[o] for o in OZELLIKLER],
                        "Ağırlık": xgb_step.feature_importances_
                    }).sort_values(by="Ağırlık", ascending=True)

                    fig_importance = px.bar(
                        importance_df,
                        x="Ağırlık",
                        y="Özellik",
                        orientation="h",
                        color="Ağırlık",
                        color_continuous_scale="tealgrn",
                        text_auto=".2f"
                    )
                    fig_importance.update_layout(
                        template="plotly_dark",
                        height=420,
                        margin=dict(l=10, r=10, t=20, b=10),
                        coloraxis_showscale=False,
                        xaxis_title="Relatif Önem Skoru",
                        yaxis_title="",
                        xaxis=dict(fixedrange=True),
                        yaxis=dict(fixedrange=True)
                    )
                    st.plotly_chart(
                        fig_importance,
                        use_container_width=True,
                        config=PLOTLY_STATIK_CONFIG
                    )

                with sub_col2:
                    st.subheader("📋 Güncel Tahmin & Doğrulama Geçmişi")

                    # Out-of-Sample test setindeki son gerçekleşmiş 4 işlem günü +
                    # henüz vadesi tamamlanmamış son 'horizon' günü birlikte gösteriyoruz:
                    pencere_boyutu = 4 + horizon
                    son_gunler_df = data.tail(pencere_boyutu).copy()
                    y_pred_serisi = pd.Series(y_pred, index=y_test.index)
                    canli_tahminler = pd.Series(
                        model.predict(son_gunler_df[OZELLIKLER]),
                        index=son_gunler_df.index
                    )

                    tablo_verileri = []
                    fiyatlar = son_gunler_df['Close'].values
                    tarihler = son_gunler_df.index

                    for i in range(len(son_gunler_df)):
                        tarih_idx = tarihler[i]
                        tarih_str = tarih_idx.strftime("%d.%m.%Y")
                        kapanis_tl = f"{fiyatlar[i]:.2f}"

                        # Out-of-Sample test setinde yer alan günler için doğrudan y_pred kullanılır
                        if tarih_idx in y_pred_serisi.index:
                            gun_tahmini = int(y_pred_serisi.loc[tarih_idx])
                        else:
                            gun_tahmini = int(canli_tahminler.loc[tarih_idx])

                        tahmin_etiketi = "Yükseliş 🟢" if gun_tahmini == 1 else "Düşüş 🔴"

                        # Henüz 'horizon' iş günü sonrasının kapanışı gerçekleşmemişse test aşamasındadır
                        if i + horizon >= len(son_gunler_df):
                            durum = "Test Ediliyor ⏳"
                        else:
                            if tarih_idx in y_test.index:
                                gercek_yon = int(y_test.loc[tarih_idx])
                            else:
                                gercek_yon = 1 if fiyatlar[i + horizon] > fiyatlar[i] else 0

                            if gun_tahmini == gercek_yon:
                                durum = "Başarılı ✅"
                            else:
                                durum = "Hatalı ❌"

                        tablo_verileri.append({
                            "Tarih": tarih_str,
                            "Kapanış (TL)": kapanis_tl,
                            "Model Tahmini": tahmin_etiketi,
                            "Gerçekleşen Durum": durum
                        })

                    gecmis_df = pd.DataFrame(tablo_verileri)
                    st.dataframe(gecmis_df, use_container_width=True, hide_index=True)

                # --- KÜMÜLATİF GETİRİ & POZİSYON KORUMALI (SWING/SIGNAL-TRANSITION) BACKTEST ---
                st.markdown("---")
                baslik_col, indir_col = st.columns([3, 1])
                with baslik_col:
                    st.subheader("💰 Kümülatif Getiri Karşılaştırması (Pozisyon Korumalı Test Seti Backtest)")

                # 1. POZİSYON TAKİBİ (POSITION HOLDING) BAZ GETİRİSİ:
                # Günlük baz getiri olarak kapanıştan kapanışa getiri kullanılır.
                gunluk_fiyat_getirisi = data['Close'].pct_change().shift(-1)

                # 2. Modelin Out-of-Sample test setindeki günlere ve tahminlere odaklan
                backtest_df = pd.DataFrame({
                    'Kapanis_Fiyati': data['Close'].loc[y_test.index],
                    'Gercek_Yon': y_test,
                    'Tahmin': y_pred,
                    'Guven_Orani': y_test_guven_serisi.loc[y_test.index],
                    'Gunluk_Fiyat_Getirisi': gunluk_fiyat_getirisi.loc[y_test.index]
                }, index=y_test.index).dropna()

                # Kullanıcının seçtiği görünüm periyoduna göre backtest penceresini filtrele
                backtest_gorunum = backtest_df[backtest_df.index >= gorunum_baslangic].copy()
                if len(backtest_gorunum) >= 2:
                    backtest_df = backtest_gorunum

                # 3. POZİSYON KORUMA VE SADECE GEÇİŞLERDE KOMİSYON KESİNTİSİ:
                # Tahmin = 1 olduğunda seçilen tahmin ufku (horizon) boyunca pozisyon korunur (Pozisyon = 1),
                # sinyal 0'a döndüğünde nakitte kalınır (Pozisyon = 0).
                backtest_df['Pozisyon'] = (
                    backtest_df['Tahmin']
                    .rolling(window=horizon, min_periods=1)
                    .max()
                    .astype(int)
                )

                # Komisyon SADECE pozisyon değiştiğinde (0 -> 1 alım veya 1 -> 0 satım) tek yönlü kesilir.
                # Üst üste pozisyonda kalınan günlerde tekrar komisyon ödenmez.
                komisyon_ondalik = komisyon_orani_yuzde / 100.0
                pozisyon_degisimi = (
                    backtest_df['Pozisyon']
                    .diff()
                    .fillna(backtest_df['Pozisyon'])
                    .abs()
                )
                backtest_df['Pozisyon_Degisimi'] = pozisyon_degisimi

                ham_strateji_getirisi = np.where(
                    backtest_df['Pozisyon'] == 1,
                    backtest_df['Gunluk_Fiyat_Getirisi'],
                    0.0
                )
                backtest_df['Model_Getiri'] = ham_strateji_getirisi - (pozisyon_degisimi * komisyon_ondalik)

                # Al-ve-Tut (Buy & Hold) ve Model Stratejisi TL Bazlı Portföy Büyümesi
                backtest_df['Pasif'] = (
                    float(baslangic_sermayesi)
                    * (1.0 - komisyon_ondalik)
                    * (1.0 + backtest_df['Gunluk_Fiyat_Getirisi']).cumprod()
                )
                backtest_df['Model'] = (
                    float(baslangic_sermayesi)
                    * (1.0 + backtest_df['Model_Getiri']).cumprod()
                )

                # 4. CSV DIŞA AKTARMA (UTF-8-SIG / Excel Uyumlu):
                if len(backtest_df) > 0:
                    export_df = pd.DataFrame({
                        "Tarih": backtest_df.index.strftime("%Y-%m-%d"),
                        "Kapanış Fiyatı (TL)": backtest_df["Kapanis_Fiyati"].round(2).values,
                        "Gerçek Yön": np.where(backtest_df["Gercek_Yon"] == 1, "Yükseliş (1)", "Düşüş (0)"),
                        "Model Tahmini": np.where(backtest_df["Tahmin"] == 1, "Yükseliş (1)", "Düşüş (0)"),
                        "Güven Oranı (%)": backtest_df["Guven_Orani"].round(2).values,
                        "Komisyon Sonrası Kümülatif Portföy Değeri (TL)": backtest_df["Model"].round(2).values,
                        "Al-Tut Kümülatif Portföy Değeri (TL)": backtest_df["Pasif"].round(2).values
                    })
                    csv_rapor_bytes = export_df.to_csv(index=False).encode("utf-8-sig")
                    temiz_kod = secilen_sembol.replace(".IS", "")

                    with indir_col:
                        st.download_button(
                            label="📥 Backtest Raporunu İndir (.CSV)",
                            data=csv_rapor_bytes,
                            file_name=f"{temiz_kod}_backtest_raporu.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                # 5. KANTİTATİF FİNANSAL PERFORMANS METRİKLERİ:
                if len(backtest_df) > 0:
                    net_portfoy_degeri = float(backtest_df['Model'].iloc[-1])
                    pasif_portfoy_degeri = float(backtest_df['Pasif'].iloc[-1])

                    model_toplam_getiri_yuzde = ((net_portfoy_degeri - baslangic_sermayesi) / baslangic_sermayesi) * 100.0
                    pasif_toplam_getiri_yuzde = ((pasif_portfoy_degeri - baslangic_sermayesi) / baslangic_sermayesi) * 100.0
                    alfa_farki = model_toplam_getiri_yuzde - pasif_toplam_getiri_yuzde

                    # Yıllıklandırılmış Sharpe Oranı (Sıfıra bölünme korumalı)
                    ort_gunluk_net = float(backtest_df['Model_Getiri'].mean())
                    std_gunluk_net = float(backtest_df['Model_Getiri'].std())
                    if std_gunluk_net > 1e-9:
                        sharpe_orani = (ort_gunluk_net / std_gunluk_net) * np.sqrt(252)
                    else:
                        sharpe_orani = 0.0

                    # Maksimum Düşüş (Max Drawdown - MDD): (Peak - Trough) / Peak
                    portfoy_serisi = pd.concat([
                        pd.Series([float(baslangic_sermayesi)]),
                        backtest_df['Model'].reset_index(drop=True)
                    ], ignore_index=True)
                    zirve_serisi = portfoy_serisi.cummax()
                    dusus_serisi = (zirve_serisi - portfoy_serisi) / zirve_serisi
                    max_drawdown_yuzde = float(dusus_serisi.max()) * 100.0

                    # Kazanma Oranı (Win Rate): Pozisyonda olunan kârlı günlerin oranı ve işlem geçiş sayısı
                    pozisyonda_gunler = backtest_df[backtest_df['Pozisyon'] == 1]
                    toplam_pozisyon_gun = int(len(pozisyonda_gunler))
                    karli_pozisyon_gun = int((pozisyonda_gunler['Model_Getiri'] > 0).sum())
                    win_rate_yuzde = (
                        (karli_pozisyon_gun / toplam_pozisyon_gun * 100.0)
                        if toplam_pozisyon_gun > 0
                        else 0.0
                    )
                    toplam_islem_gecisi = int(backtest_df['Pozisyon_Degisimi'].sum())

                    # 4'lü Finansal Performans Kartları (Mobil Uyumlu Kısa Format)
                    bt_col1, bt_col2, bt_col3, bt_col4 = st.columns(4)
                    bt_col1.metric(
                        "Net Portföy & Toplam Getiri",
                        f"{net_portfoy_degeri:,.0f} TL",
                        delta=f"{model_toplam_getiri_yuzde:+.1f}% (Al-Tut'a göre {alfa_farki:+.1f}%)"
                    )
                    bt_col2.metric(
                        "Sharpe Oranı (Yıllık)",
                        f"{sharpe_orani:.2f}",
                        delta="Risk-Ayarlı Getiri" if sharpe_orani >= 1.0 else "Dalgalı Getiri",
                        delta_color="normal" if sharpe_orani >= 0 else "inverse"
                    )
                    bt_col3.metric(
                        "Maksimum Düşüş (MDD)",
                        f"-%{max_drawdown_yuzde:.2f}",
                        delta="Zirveden En Büyük Kayıp",
                        delta_color="inverse"
                    )
                    bt_col4.metric(
                        "Kazanma Oranı (Win Rate)",
                        f"%{win_rate_yuzde:.1f}",
                        delta=f"{karli_pozisyon_gun}/{toplam_pozisyon_gun} Kârlı Gün ({toplam_islem_gecisi} İşlem)"
                    )

                # 6. Plotly çizgi grafiği (15 Günlük Sağ Analist Boşluğu ve Sınırlandırılmış X/Y Eksenleri ile)
                bt_alt_sinir = float(min(backtest_df['Pasif'].min(), backtest_df['Model'].min())) * 0.95
                bt_ust_sinir = float(max(backtest_df['Pasif'].max(), backtest_df['Model'].max())) * 1.05
                bt_x_baslangic = backtest_df.index[0]
                bt_x_bitis_bosluklu = backtest_df.index[-1] + pd.Timedelta(days=15)

                fig_backtest = go.Figure()
                fig_backtest.add_trace(go.Scatter(
                    x=backtest_df.index, y=backtest_df['Pasif'],
                    mode='lines', name='Pasif Getiri (Al-Tut)',
                    line=dict(color='#9e9e9e', width=2, dash='dash')
                ))
                fig_backtest.add_trace(go.Scatter(
                    x=backtest_df.index, y=backtest_df['Model'],
                    mode='lines', name='Model Stratejisi (Pozisyon Korumalı)',
                    line=dict(color='#26a69a', width=3)
                ))
                fig_backtest.update_layout(
                    template="plotly_dark",
                    height=380,
                    dragmode="pan",
                    margin=dict(l=10, r=10, t=20, b=10),
                    xaxis_title="Tarih",
                    yaxis_title=f"Portföy Değeri (TL, Başlangıç = {baslangic_sermayesi:,.0f} TL)",
                    xaxis=dict(
                        nticks=6,
                        tickformat="%d %b '%y",
                        tickangle=0,
                        autorange=False,
                        range=[bt_x_baslangic, bt_x_bitis_bosluklu],
                        minallowed=bt_x_baslangic,
                        maxallowed=bt_x_bitis_bosluklu,
                        fixedrange=False
                    ),
                    yaxis=dict(
                        autorange=False,
                        range=[bt_alt_sinir, bt_ust_sinir],
                        minallowed=bt_alt_sinir,
                        maxallowed=bt_ust_sinir,
                        fixedrange=True
                    ),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                    hovermode="x unified"
                )
                st.info(
                    "Nasıl Okunmalı: Gri kesikli çizgi, başlangıç sermayenizle hisseyi alıp hiç satmasaydınız (Al-Tut) "
                    "portföyünüzün durumunu; yeşil çizgi, model Yükseliş sinyali verdiğinde hisseye girip sinyal "
                    "Düşüşe dönene kadar pozisyonu koruyan (yalnızca alım-satım geçişlerinde komisyon ödeyen) "
                    "akıllı stratejinin net portföy büyüklüğünü gösterir."
                )
                st.plotly_chart(
                    fig_backtest,
                    use_container_width=True,
                    config={"displayModeBar": False, "scrollZoom": True, "doubleClick": "reset"}
                )

                if len(backtest_df) > 0:
                    st.caption(
                        f"Simülasyon Özeti ({len(backtest_df)} işlem günü | Başlangıç: **{baslangic_sermayesi:,.0f} TL** | "
                        f"Tek Yön Komisyon: **%{komisyon_orani_yuzde:.2f}**): "
                        f"Al-ve-Tut **{backtest_df['Pasif'].iloc[-1]:,.2f} TL** | "
                        f"Model Stratejisi **{backtest_df['Model'].iloc[-1]:,.2f} TL**. "
                        "Geçmiş performans gelecek getirinin garantisi değildir."
                    )

            # 3. SEKME: İstatistiksel Metrikler (Out-of-Sample test seti üzerinden)
            with sekme_istatistik:
                st.info(
                    "Araştırmacı Notu: Bu sekmedeki metrikler (Sınıflandırma Raporu ve Karmaşıklık Matrisi), "
                    "modelin teknik davranışını, sınıf dengesizliklerini ve hata dağılımlarını analiz etmek içindir "
                    "(Laboratuvar Testi). Günlük alım-satım kararlarında modelin gerçek dünya performansını görmek "
                    "için lütfen Fiyat ve Tahmin sekmesindeki Walk-Forward CV (İleriye Yönelik Doğrulama) oranını referans alınız."
                )
                st.caption(
                    f"Değerlendirme, modelin eğitimde hiç görmediği kronolojik son %20'lik test seti "
                    f"({len(y_test)} işlem günü | Tahmin Ufku: {secilen_ufuk_str}) üzerinden yapılmıştır."
                )

                metrik_col1, metrik_col2 = st.columns(2)

                # --- A) Sınıflandırma Raporu (Tam Türkçeleştirilmiş) ---
                with metrik_col1:
                    st.subheader("📑 Sınıflandırma Raporu")
                    rapor_dict = classification_report(
                        y_test, y_pred,
                        labels=[0, 1],
                        target_names=["Düşüş (0)", "Yükseliş (1)"],
                        output_dict=True,
                        zero_division=0
                    )
                    dogruluk_degeri = float(rapor_dict.get("accuracy", accuracy_score(y_test, y_pred)))
                    rapor_dict["accuracy"] = {
                        "precision": dogruluk_degeri,
                        "recall": dogruluk_degeri,
                        "f1-score": dogruluk_degeri,
                        "support": len(y_test)
                    }

                    rapor_df = pd.DataFrame(rapor_dict).T
                    rapor_df = rapor_df.rename(index={
                        "0": "Düşüş (0)",
                        "Düşüş": "Düşüş (0)",
                        "1": "Yükseliş (1)",
                        "Yükseliş": "Yükseliş (1)",
                        "accuracy": "Genel Doğruluk",
                        "macro avg": "Makro Ortalama",
                        "weighted avg": "Ağırlıklı Ortalama"
                    })
                    rapor_df = rapor_df.rename(columns={
                        "precision": "Kesinlik (Precision)",
                        "recall": "Duyarlılık (Recall)",
                        "f1-score": "F1-Skoru",
                        "support": "Örneklem Sayısı (Gün)"
                    })
                    rapor_df["Örneklem Sayısı (Gün)"] = rapor_df["Örneklem Sayısı (Gün)"].astype(int)
                    st.dataframe(
                        rapor_df.style.format({
                            "Kesinlik (Precision)": "{:.2f}",
                            "Duyarlılık (Recall)": "{:.2f}",
                            "F1-Skoru": "{:.2f}"
                        }),
                        use_container_width=True
                    )
                    st.caption(f"Test Seti Genel Doğruluk (Accuracy): **%{dogruluk_degeri*100:.2f}**")

                # --- B) Karmaşıklık Matrisi (Confusion Matrix) ---
                with metrik_col2:
                    st.subheader("🧩 Karmaşıklık Matrisi")
                    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
                    etiketler = ["Düşüş (0)", "Yükseliş (1)"]

                    fig_cm = go.Figure(data=go.Heatmap(
                        z=cm,
                        x=etiketler,
                        y=etiketler,
                        colorscale="Tealgrn",
                        showscale=False,
                        text=cm,
                        texttemplate="%{text}",
                        textfont={"size": 24},
                        hovertemplate="Gerçek: %{y}<br>Tahmin: %{x}<br>Adet: %{z}<extra></extra>"
                    ))
                    fig_cm.update_layout(
                        template="plotly_dark",
                        height=340,
                        margin=dict(l=10, r=10, t=20, b=10),
                        xaxis_title="Tahmin Edilen Değer",
                        yaxis_title="Gerçek Değer",
                        xaxis=dict(fixedrange=True),
                        yaxis=dict(autorange="reversed", fixedrange=True)
                    )
                    st.plotly_chart(
                        fig_cm,
                        use_container_width=True,
                        config=PLOTLY_STATIK_CONFIG
                    )

            # 4. SEKME: Hızlı BIST Sinyal Tarayıcı (Batch Screener)
            with sekme_tarayici:
                st.subheader(f"🔍 BIST Toplu Sinyal Tarayıcı ({secilen_ufuk_str})")
                st.caption(
                    "Performans optimizasyonu kapsamında varsayılan olarak BIST 30'un en likit 10 lokomotif hissesi "
                    "seçilidir. Listeye hisse ekleyip çıkarabilirsiniz (Maksimum 15 hisse)."
                )

                with st.form("screener_form"):
                    secili_tarama_hisseleri = st.multiselect(
                        "Taranacak Hisseleri Seçin (Maksimum 15):",
                        options=list(BIST_HISSELERI.keys()),
                        default=VARSAYILAN_TARAMA_HISSELERI,
                        max_selections=15,
                        format_func=lambda kod: f"{kod} - {BIST_HISSELERI.get(kod, kod)}"
                    )

                    tara_butonu = st.form_submit_button(
                        "🚀 Seçili Hisseleri Tara",
                        type="primary",
                        use_container_width=True
                    )

                if tara_butonu:
                    if not secili_tarama_hisseleri:
                        st.warning("Lütfen tarama yapmak için en az 1 hisse seçiniz.")
                    else:
                        if len(secili_tarama_hisseleri) > 15:
                            st.warning("Performans sınırı nedeniyle ilk 15 hisse taranacaktır.")
                            secili_tarama_hisseleri = secili_tarama_hisseleri[:15]

                        ilerleme_cubugu = st.progress(0, text="Hisseler taranıyor, lütfen bekleyin...")
                        tarama_sonuclari = []
                        toplam_hisse = len(secili_tarama_hisseleri)

                        for idx, hisse_kodu in enumerate(secili_tarama_hisseleri):
                            ilerleme_cubugu.progress(
                                (idx) / toplam_hisse,
                                text=f"🔄 {hisse_kodu} ({idx + 1}/{toplam_hisse}) analiz ediliyor..."
                            )
                            sonuc = hizli_sinyal_tara(hisse_kodu, horizon=horizon)
                            if sonuc is not None:
                                tarama_sonuclari.append(sonuc)
                            ilerleme_cubugu.progress(
                                (idx + 1) / toplam_hisse,
                                text=f"✅ {hisse_kodu} ({idx + 1}/{toplam_hisse}) tamamlandı."
                            )

                        ilerleme_cubugu.empty()

                        if tarama_sonuclari:
                            sonuc_df = pd.DataFrame(tarama_sonuclari).sort_values(
                                by="Tahmin Güven Oranı (%)", ascending=False
                            ).reset_index(drop=True)
                            st.session_state.screener_sonuc_df = sonuc_df
                            st.session_state.screener_ufuk = horizon
                        else:
                            st.error("Seçili hisseler için tarama sonucu üretilemedi.")

                if st.session_state.screener_sonuc_df is not None:
                    if st.session_state.screener_ufuk != horizon:
                        st.warning(
                            "Tahmin ufku değiştirildi. Yeni ufka göre güncel sinyalleri görmek için "
                            "lütfen 'Seçili Hisseleri Tara' butonuna tekrar basın."
                        )

                    gosterilecek_df = st.session_state.screener_sonuc_df

                    # Günün En Güçlü Yükseliş ve Düşüş Adayı Özet Kutuları
                    yukselis_adaylari = gosterilecek_df[gosterilecek_df["Sinyal"] == "🟢 YÜKSELİŞ"]
                    dusus_adaylari = gosterilecek_df[gosterilecek_df["Sinyal"] == "🔴 DÜŞÜŞ"]

                    aday_col1, aday_col2 = st.columns(2)
                    with aday_col1:
                        if not yukselis_adaylari.empty:
                            en_guclu_yukselis = yukselis_adaylari.iloc[0]
                            st.success(
                                f"🚀 **Günün En Güçlü Yükseliş Adayı:** "
                                f"**{en_guclu_yukselis['Hisse Kodu']} ({en_guclu_yukselis['Şirket Adı']})**  \n"
                                f"Güven Oranı: **%{en_guclu_yukselis['Tahmin Güven Oranı (%)']:.2f}** | "
                                f"CV Doğruluğu: **%{en_guclu_yukselis['Walk-Forward CV Doğruluğu (%)']:.2f}** | "
                                f"Fiyat: **{en_guclu_yukselis['Son Fiyat (TL)']:.2f} TL**"
                            )
                        else:
                            st.info("Taranan hisseler arasında aktif Yükseliş sinyali bulunmuyor.")

                    with aday_col2:
                        if not dusus_adaylari.empty:
                            en_guclu_dusus = dusus_adaylari.iloc[0]
                            st.error(
                                f"📉 **Günün En Güçlü Düşüş Adayı:** "
                                f"**{en_guclu_dusus['Hisse Kodu']} ({en_guclu_dusus['Şirket Adı']})**  \n"
                                f"Güven Oranı: **%{en_guclu_dusus['Tahmin Güven Oranı (%)']:.2f}** | "
                                f"CV Doğruluğu: **%{en_guclu_dusus['Walk-Forward CV Doğruluğu (%)']:.2f}** | "
                                f"Fiyat: **{en_guclu_dusus['Son Fiyat (TL)']:.2f} TL**"
                            )
                        else:
                            st.info("Taranan hisseler arasında aktif Düşüş sinyali bulunmuyor.")

                    st.dataframe(
                        gosterilecek_df.style.format({
                            "Son Fiyat (TL)": "{:.2f}",
                            "Tahmin Güven Oranı (%)": "%{:.2f}",
                            "Walk-Forward CV Doğruluğu (%)": "%{:.2f}",
                            "Güncel RSI (14)": "{:.1f}"
                        }),
                        use_container_width=True,
                        hide_index=True
                    )

                    screener_csv_bytes = gosterilecek_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="📥 Toplu Sinyal Tarama Sonuçlarını İndir (bist_gunluk_sinyal_taramasi.csv)",
                        data=screener_csv_bytes,
                        file_name="bist_gunluk_sinyal_taramasi.csv",
                        mime="text/csv"
                    )

            # 5. SEKME: Haber Akışı (Şirket Tam Adı + Sembol Araması ve Türkçe Tarih Formatı)
            with sekme_haber:
                temiz_hisse_adi = secilen_sembol.replace(".IS", "")
                sirket_tam_adi = BIST_HISSELERI.get(secilen_sembol, temiz_hisse_adi)
                st.subheader(f"📰 {sirket_tam_adi} ({temiz_hisse_adi}) Güncel Haber Akışı")

                arama_sorgusu = f"{sirket_tam_adi} {temiz_hisse_adi} hisse"
                rss_url = f"https://news.google.com/rss/search?q={urllib.parse.quote(arama_sorgusu)}&hl=tr&gl=TR&ceid=TR:tr"

                ayristirilmis_haberler = []
                try:
                    istek = urllib.request.Request(
                        rss_url,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                    with urllib.request.urlopen(istek, timeout=10) as yanit:
                        xml_verisi = yanit.read()

                    kok = ET.fromstring(xml_verisi)
                    tum_ogeler = kok.findall(".//item")

                    for item in tum_ogeler:
                        baslik = item.findtext("title", default="Başlık Yok")
                        link = item.findtext("link", default="#")
                        ham_tarih = item.findtext("pubDate", default="")
                        kaynak = item.findtext("source", default="Google Haberler")
                        dt_obj, tr_tarih_str = rss_tarih_parse_ve_formatla(ham_tarih)
                        ayristirilmis_haberler.append({
                            "baslik": baslik,
                            "link": link,
                            "kaynak": kaynak,
                            "dt": dt_obj,
                            "tarih_str": tr_tarih_str
                        })

                    # Haberleri en yeniden en eskiye doğru sırala ve ilk 5 haberi al
                    ayristirilmis_haberler.sort(key=lambda x: x["dt"], reverse=True)
                    ayristirilmis_haberler = ayristirilmis_haberler[:5]
                except Exception:
                    ayristirilmis_haberler = []

                if ayristirilmis_haberler:
                    for haber in ayristirilmis_haberler:
                        with st.container(border=True):
                            st.markdown(f"#### [{haber['baslik']}]({haber['link']})")
                            st.caption(
                                f"🏢 **Kaynak:** {haber['kaynak']} &nbsp;|&nbsp; "
                                f"🕒 **Yayınlanma Tarihi:** {haber['tarih_str']}"
                            )
                else:
                    st.warning("Bu hisse için güncel haber bulunamadı.")

        else:
            st.error("Veri çekme hatası oluştu veya yeterli geçmiş veri bulunamadı.")

else:
    st.info(f"Lütfen bir hisse seçin ve 'Analizi Başlat' butonuna tıklayın. (Şu an seçili: {secilen_hisse_str})")

st.markdown("---")
st.caption("⚖️ **Yasal Uyarı:** Bu panel akademik bir istatistiksel modelleme projesi olup, üretilen tahminler kesinlikle yatırım tavsiyesi (YTD) değildir.")

st.sidebar.markdown("---")
st.sidebar.caption("Sürüm: 3.0 | Dinamik Tahmin Ufku, BIST Sinyal Tarayıcı & Kantitatif Backtest")