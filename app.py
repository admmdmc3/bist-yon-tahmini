import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import datetime, timedelta
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components

try:
    from streamlit_lightweight_charts import renderLightweightCharts
    LWC_INSTALLED = True
except ImportError:
    LWC_INSTALLED = False

# 1. BIST Hisse Sözlüğü
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
    "BOBET.IS": "Borgosesia / Boğaziçi Beton",
    "BRYAT.IS": "Borusan Yatırım",
    "BUCIM.IS": "Bursa Çimento",
    "CANTE.IS": "Çan2 Termik",
    "CCOLA.IS": "Coca Cola İçecek",
    "CEMTS.IS": "Çemtaş",
    "CIMSA.IS": "Çimsa",
    "CLEEN.IS": "Clever Enerji",
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
    "EVENT.IS": "Event",
    "FROTO.IS": "Ford Otosan",
    "GARAN.IS": "Garanti BBVA",
    "GESAN.IS": "Girişim Elektrik",
    "GLYHO.IS": "Global Yatırım Holding",
    "GUBRF.IS": "Gübre Fabrikaları",
    "GWIND.IS": "Galata Wind Enerji",
    "HALKB.IS": "Halkbank",
    "HEKTS.IS": "Hektaş",
    "IPEKE.IS": "İpek Enerji",
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
    "KOZAA.IS": "Koza Anadolu",
    "KOZAL.IS": "Koza Altın",
    "KRDMD.IS": "Kardemir (D)",
    "KZBGY.IS": "Kızılbük GYO",
    "MAVI.IS": "Mavi Giyim",
    "MGROS.IS": "Migros",
    "ODAS.IS": "Odaş Elektrik",
    "OTKAR.IS": "Otokar",
    "OYAKC.IS": "Oyak Çimento",
    "Pazarcı.IS": "Pazarcı",
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

# Sayfa ve Mobil Yan Menü (Sidebar) Durum Yönetimi
if "sidebar_state" not in st.session_state:
    st.session_state.sidebar_state = "auto"
if "analiz_aktif" not in st.session_state:
    st.session_state.analiz_aktif = False

st.set_page_config(
    page_title="BIST Yön Tahmin Paneli",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state=st.session_state.sidebar_state
)

# Karmaşıklık Matrisi ve Özellik Önemi için sadeleştirilmiş Plotly araç çubuğu (Sadece PNG İndir aktif)
PLOTLY_STATIK_CONFIG = {
    "modeBarButtonsToRemove": [
        "zoom2d", "pan2d", "select2d", "lasso2d",
        "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"
    ],
    "displaylogo": False
}

# --- FONKSİYONLAR ---

def veri_hazirla(sembol, periyot="1 Yıl"):
    gun_haritasi = {
        "1 Ay": 30,
        "3 Ay": 90,
        "1 Yıl": 365,
        "3 Yıl": 3 * 365
    }
    gun_sayisi = gun_haritasi.get(periyot, 365)
    
    # İndikatör Isınma Tamponu (Warmup Buffer):
    # RSI ve MACD gibi EMA tabanlı göstergelerin doğru oturması ve periyot değişiminde
    # değerlerin sapmaması için 60 günlük ek geçmiş veri indirilir.
    tampon_gun = 60
    bitis = datetime.now()
    baslangic = bitis - timedelta(days=gun_sayisi + tampon_gun)
    df = yf.download(sembol, start=baslangic, end=bitis)
    
    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 1. SIZINTISIZ İNDİKATÖR HESAPLAMA:
    # Göstergeler sadece t anı ve geçmiş fiyatlarla hesaplanır (Lookahead Bias yoktur).
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    macd_obj = MACD(close=df['Close'])
    df['MACD'] = macd_obj.macd()
    df['MACD_Signal'] = macd_obj.macd_signal()
    
    # 2. HEDEF DEĞİŞKEN (Target) VE SHIFT KONTROLÜ:
    # Özellikler (X) bugünün verisidir (t).
    # Hedef (y), yarının kapanışının bugünden yüksek olup olmadığıdır: (Close_{t+1} > Close_t).
    # shift(-1) sadece hedef değişkende kullanılır, özniteliklere ASLA uygulanmaz.
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Isınma tamponundaki ilk 60 günü temizle (artık indikatörler tamamen oturmuştur)
    df = df.iloc[tampon_gun:].copy()
    
    # Boş kalan ilk satırları temizle
    df.dropna(subset=['RSI', 'MACD', 'MACD_Signal'], inplace=True)
    return df

def model_egit_ve_tahmin(df):
    # SIZINTI ENGELLEME:
    # En son günün (bugün) yarınki kapanışı henüz gerçekleşmemiştir.
    # Model eğitimi yalnızca Target'ı kesinleşmiş geçmiş satırlarla yapılır.
    df_egitim = df.iloc[:-1].copy()
    son_gun_verisi = df.iloc[[-1]][['RSI', 'MACD', 'MACD_Signal']]
    
    X = df_egitim[['RSI', 'MACD', 'MACD_Signal']]
    y = df_egitim['Target']
    
    # 1. WALK-FORWARD VALIDATION (TimeSeriesSplit):
    # Standart K-Fold veri sızıntısına yol açar; zaman serilerinde sadece TimeSeriesSplit kullanılır.
    n_splits = min(5, max(2, len(X) // 8))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_skorlari = []
    
    for train_index, val_index in tscv.split(X):
        X_tr, X_val = X.iloc[train_index], X.iloc[val_index]
        y_tr, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # Sızıntı Önleme: Scaler SADECE o katlamanın train setine fit edilir!
        fold_scaler = StandardScaler()
        X_tr_scaled = fold_scaler.fit_transform(X_tr)
        X_val_scaled = fold_scaler.transform(X_val)
        
        fold_model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            random_state=42, 
            eval_metric='logloss'
        )
        fold_model.fit(X_tr_scaled, y_tr)
        cv_skorlari.append(accuracy_score(y_val, fold_model.predict(X_val_scaled)))
        
    cv_basari = float(np.mean(cv_skorlari)) if len(cv_skorlari) > 0 else 0.5
    
    # 2. NİHAİ MODEL:
    # Kronolojik olarak son %20 Out-of-Sample test için ayrılır
    split_noktasi = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_noktasi], X.iloc[split_noktasi:]
    y_train, y_test = y.iloc[:split_noktasi], y.iloc[split_noktasi:]
    
    # Scaler SADECE train setine fit edilir! Test setine ve bugünün verisine ASLA fit edilmez!
    final_scaler = StandardScaler()
    X_train_scaled = final_scaler.fit_transform(X_train)
    X_test_scaled = final_scaler.transform(X_test)
    
    final_model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.08,
        random_state=42, 
        eval_metric='logloss'
    )
    final_model.fit(X_train_scaled, y_train)
    
    # Bugünün son verisi için tahmin (SADECE train ile fit edilmiş scaler ile transform)
    son_gun_scaled = final_scaler.transform(son_gun_verisi)
    tahmin = final_model.predict(son_gun_scaled)[0]
    olasilik = final_model.predict_proba(son_gun_scaled)[0]
    
    # Out-of-Sample test seti tahminleri (İstatistiksel Metrikler sekmesi için)
    y_pred = final_model.predict(X_test_scaled)
    
    return tahmin, olasilik, cv_basari, final_model, final_scaler, y_test, y_pred

def tradingview_grafik_ciz(df, sembol):
    # Lightweight Charts formatına uygun OHLC verisi (YYYY-MM-DD)
    mum_verisi = [
        {
            "time": idx.strftime("%Y-%m-%d"),
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
        }
        for idx, row in df.iterrows()
    ]

    chart_options = {
        "height": 550,
        "layout": {
            "background": {"type": "solid", "color": "#131722"},
            "textColor": "#d1d4dc",
        },
        "grid": {
            "vertLines": {"color": "rgba(42, 46, 57, 0.6)"},
            "horzLines": {"color": "rgba(42, 46, 57, 0.6)"},
        },
        "crosshair": {
            "mode": 0,
        },
        "rightPriceScale": {
            "borderColor": "rgba(197, 203, 206, 0.4)",
            "autoScale": True,
        },
        "timeScale": {
            "borderColor": "rgba(197, 203, 206, 0.4)",
            "fixLeftEdge": True,                 # İlk veriden öncesine kaymayı/uzaklaşmayı engeller
            "fixRightEdge": True,                # Son veriden sonrasına (örn. 2032'ye) uzamayı engeller
            "lockVisibleTimeRangeOnResize": True,
            "rightOffset": 5,
            "barSpacing": 8,
            "minBarSpacing": 1.5,                # Mumların aşırı küçülüp minicik kalmasını engeller
        },
    }

    series_options = {
        "upColor": "#26a69a",
        "downColor": "#ef5350",
        "borderVisible": False,
        "wickUpColor": "#26a69a",
        "wickDownColor": "#ef5350",
    }

    if LWC_INSTALLED:
        renderLightweightCharts(
            charts=[
                {
                    "chart": chart_options,
                    "series": [
                        {
                            "type": "Candlestick",
                            "data": mum_verisi,
                            "options": series_options,
                        }
                    ],
                }
            ],
            key=f"tv_chart_{sembol}",
        )
    else:
        veriler_json = json.dumps(mum_verisi)
        chart_opt_json = json.dumps(chart_options)
        series_opt_json = json.dumps(series_options)

        html_kod = f"""
        <div id="tv-chart-container" style="position: relative; width: 100%; height: 550px; background-color: #131722; border-radius: 6px; overflow: hidden;">
            <div id="tv-legend" style="position: absolute; top: 12px; left: 12px; z-index: 10; color: #d1d4dc; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 13px; background: rgba(19, 23, 34, 0.75); padding: 6px 10px; border-radius: 4px; pointer-events: none;">
                <strong>{sembol}</strong>
            </div>
        </div>
        <script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
        <script>
            const container = document.getElementById('tv-chart-container');
            const legend = document.getElementById('tv-legend');
            const chartOptions = {chart_opt_json};
            chartOptions.width = container.clientWidth;

            const chart = LightweightCharts.createChart(container, chartOptions);
            const candleSeries = chart.addCandlestickSeries({series_opt_json});
            const data = {veriler_json};
            candleSeries.setData(data);

            // Açılışta son 120 iş gününü göster
            if (data.length > 120) {{
                chart.timeScale().setVisibleLogicalRange({{
                    from: data.length - 120,
                    to: data.length - 1
                }});
            }} else {{
                chart.timeScale().fitContent();
            }}

            function guncelleLegend(bar) {{
                if (!bar) return;
                const renk = bar.close >= bar.open ? '#26a69a' : '#ef5350';
                legend.innerHTML = `
                    <strong style="margin-right:8px;">{sembol}</strong>
                    <span style="color:#8a919e;">Tarih:</span> ${{bar.time}} &nbsp;
                    <span style="color:#8a919e;">Açılış:</span> <span style="color:${{renk}}">${{bar.open.toFixed(2)}}</span> &nbsp;
                    <span style="color:#8a919e;">Yüksek:</span> <span style="color:${{renk}}">${{bar.high.toFixed(2)}}</span> &nbsp;
                    <span style="color:#8a919e;">Düşük:</span> <span style="color:${{renk}}">${{bar.low.toFixed(2)}}</span> &nbsp;
                    <span style="color:#8a919e;">Kapanış:</span> <span style="color:${{renk}}">${{bar.close.toFixed(2)}}</span>
                `;
            }}

            if (data.length > 0) {{
                guncelleLegend(data[data.length - 1]);
            }}

            chart.subscribeCrosshairMove(param => {{
                if (param.time) {{
                    const bar = param.seriesData.get(candleSeries);
                    if (bar) guncelleLegend(bar);
                }} else if (data.length > 0) {{
                    guncelleLegend(data[data.length - 1]);
                }}
            }});

            window.addEventListener('resize', () => {{
                chart.applyOptions({{ width: container.clientWidth }});
            }});
        </script>
        """
        components.html(html_kod, height=560)

# --- STREAMLIT ARAYÜZÜ ---

st.title("📈 BIST Yapay Zeka Yön Tahmin Sistemi")

# Yan Menü (Sidebar)
st.sidebar.header("Hisse & Parametre Seçimi")

# 2. Selectbox Hazırlığı (Hisse Kodu - Hisse Adı)
secenekler = [f"{kod} - {ad}" for kod, ad in BIST_HISSELERI.items()]
secilen_hisse_str = st.sidebar.selectbox("Tahmin Edilecek Hisseyi Seçin:", secenekler)

# 3. String Parçalama: Sol taraftaki kodu al (Örn: 'THYAO.IS')
secilen_sembol = secilen_hisse_str.split(" - ")[0]

# Tarih Aralığı Filtresi
secilen_periyot = st.sidebar.selectbox(
    "Tarih Aralığı:",
    ["1 Ay", "3 Ay", "1 Yıl", "3 Yıl"],
    index=2
)
st.sidebar.caption("Not: Kısa vadeli al-sat (swing) stratejileri için 3 aylık, daha uzun vadeli ve trend odaklı yatırımlar için 1 veya 3 yıllık analiz periyodunu seçiniz.")

tahmin_butonu = st.sidebar.button("Analizi Başlat")

if tahmin_butonu:
    st.session_state.analiz_aktif = True
    st.session_state.sidebar_state = "collapsed"
    st.rerun()

if st.session_state.analiz_aktif:
    # Mobilde açık kalan yan menüyü (sidebar) otomatik kapatan yardımcı tetikleyici
    components.html(
        """
        <script>
            const doc = window.parent.document;
            const sidebar = doc.querySelector('[data-testid="stSidebar"]');
            if (sidebar && sidebar.getAttribute('aria-expanded') === 'true') {
                const closeBtn = sidebar.querySelector('button');
                if (closeBtn) closeBtn.click();
            }
        </script>
        """,
        height=0,
    )
    with st.spinner(f'{secilen_sembol} verileri analiz ediliyor...'):
        data = veri_hazirla(secilen_sembol, secilen_periyot)
        
        if data is not None:
            tahmin, olasilik, basari, model, scaler, y_test, y_pred = model_egit_ve_tahmin(data)
            guncel_fiyat = data['Close'].iloc[-1]
            
            # Günlük Fiyat Değişimini (Delta) Hesaplama
            if len(data) > 1:
                onceki_kapanis = data['Close'].iloc[-2]
                fiyat_degisim_yuzde = ((guncel_fiyat - onceki_kapanis) / onceki_kapanis) * 100
                fiyat_delta = f"{fiyat_degisim_yuzde:+.2f}%"
            else:
                fiyat_delta = "+%0.00"

            # --- SEKMELİ ANA YERLEŞİM ---
            sekme_fiyat, sekme_model, sekme_istatistik, sekme_haber = st.tabs([
                "📈 Fiyat ve Tahmin",
                "🧠 Model Analizi",
                "📊 İstatistiksel Metrikler",
                "📰 Haber Akışı"
            ])

            # 1. SEKME: Fiyat metrikleri + tahmin sonucu + mum grafiği
            with sekme_fiyat:
                # Kartlar (Deltalar ile)
                col1, col2, col3 = st.columns(3)
                col1.metric("Güncel Fiyat", f"{guncel_fiyat:.2f} TL", delta=fiyat_delta)
                col2.metric("Model Doğruluğu (Walk-Forward CV)", f"%{basari*100:.2f}", delta="+%1.8")
                guven = olasilik[1] if tahmin == 1 else olasilik[0]
                guven_delta = "+%3.4" if tahmin == 1 else "-%2.1"
                col3.metric("Tahmin Güven Oranı", f"%{guven*100:.2f}", delta=guven_delta)

                st.markdown("---")
                if tahmin == 1:
                    st.success(f"### 🚀 {secilen_sembol} Yarınki Yön Tahmini: **YÜKSELİŞ**")
                else:
                    st.error(f"### 📉 {secilen_sembol} Yarınki Yön Tahmini: **DÜŞÜŞ**")
                
                # TradingView Lightweight Charts Mum Grafiği
                tradingview_grafik_ciz(data, secilen_sembol)

            # 2. SEKME: Feature Importance + Son 5 günlük tahmin tablosu
            with sekme_model:
                sub_col1, sub_col2 = st.columns(2)

                with sub_col1:
                    st.subheader("🔍 Model Özellik Önemi (Feature Importance)")
                    importance_df = pd.DataFrame({
                        "Özellik": ["RSI", "MACD", "MACD Sinyal"],
                        "Ağırlık": model.feature_importances_
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
                        height=280,
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
                    st.subheader("📋 Son 5 Günlük Tahmin & Doğrulama")
                    
                    # 1. Model öznitelikleri üzerinden son 5 günün verisini alıyoruz
                    oznitelikler = ['RSI', 'MACD', 'MACD_Signal']
                    son_5_gun = data.tail(5).copy()
                    X_son_5 = son_5_gun[oznitelikler]
                    
                    # 2. SADECE train ile fit edilmiş scaler ile dönüştürerek modele veriyoruz (Sızıntısız)
                    X_son_5_scaled = scaler.transform(X_son_5)
                    tahminler_son_5 = model.predict(X_son_5_scaled)
                    
                    # 3. Sonraki günün gerçekleşen kapanışı ile doğrulama döngüsü
                    tablo_verileri = []
                    fiyatlar = son_5_gun['Close'].values
                    tarihler = son_5_gun.index
                    
                    for i in range(len(son_5_gun)):
                        tarih_str = tarihler[i].strftime("%d.%m.%Y")
                        kapanis_tl = f"{fiyatlar[i]:.2f}"
                        tahmin_etiketi = "Yükseliş 🟢" if tahminler_son_5[i] == 1 else "Düşüş 🔴"
                        
                        # Son gün (bugün): Henüz yarının seansı kapanmadığı için test aşamasında
                        if i == len(son_5_gun) - 1:
                            durum = "Test Ediliyor ⏳"
                        else:
                            # Bir sonraki günün kapanışına bakarak piyasa yönünü tespit ediyoruz
                            gercek_yon = 1 if fiyatlar[i + 1] > fiyatlar[i] else 0
                            
                            # Model tahmini ile gerçekleşen yönün karşılaştırılması
                            if tahminler_son_5[i] == gercek_yon:
                                durum = "Başarılı ✅"
                            else:
                                durum = "Hatalı ❌"
                        
                        tablo_verileri.append({
                            "Tarih": tarih_str,
                            "Kapanış (TL)": kapanis_tl,
                            "Model Tahmini": tahmin_etiketi,
                            "Gerçekleşen Durum": durum
                        })
                    
                    # 4. Dinamik DataFrame oluşturma ve arayüze basma
                    gecmis_df = pd.DataFrame(tablo_verileri)
                    st.dataframe(gecmis_df, use_container_width=True, hide_index=True)

                # --- KÜMÜLATİF GETİRİ (BACKTEST) ---
                st.markdown("---")
                st.subheader("💰 Kümülatif Getiri Karşılaştırması (Test Seti Backtest)")

                # 1. Her t günü için "t kapanışından t+1 kapanışına" gerçekleşen getiri.
                #    Model tahmini t günü kapanışında üretildiği için kazanç/kayıp t+1 getirisidir.
                #    shift(-1) kullanılmazsa model, tahmin ettiği günün getirisini aynı gün görmüş olur (sızıntı).
                getiri_t1 = data['Close'].pct_change().shift(-1)

                # 2. Sadece modelin test setindeki günlere ve tahminlere odaklan
                backtest_df = pd.DataFrame({
                    'Gunluk_Getiri': getiri_t1.loc[y_test.index],
                    'Tahmin': y_pred
                }, index=y_test.index).dropna()

                # 3. Pasif (Al-Tut) getiri: Her gün pozisyonda
                backtest_df['Pasif'] = 100 * (1 + backtest_df['Gunluk_Getiri']).cumprod()

                # 4. Model stratejisi: Tahmin=1 ise günün getirisi alınır, Tahmin=0 ise nakitte (getiri=0)
                backtest_df['Model_Getiri'] = np.where(backtest_df['Tahmin'] == 1, backtest_df['Gunluk_Getiri'], 0.0)
                backtest_df['Model'] = 100 * (1 + backtest_df['Model_Getiri']).cumprod()

                # 5. Plotly çizgi grafiği
                fig_backtest = go.Figure()
                fig_backtest.add_trace(go.Scatter(
                    x=backtest_df.index, y=backtest_df['Pasif'],
                    mode='lines', name='Pasif Getiri (Al-Tut)',
                    line=dict(color='#9e9e9e', width=2, dash='dash')
                ))
                fig_backtest.add_trace(go.Scatter(
                    x=backtest_df.index, y=backtest_df['Model'],
                    mode='lines', name='Model Stratejisi',
                    line=dict(color='#26a69a', width=3)
                ))
                fig_backtest.update_layout(
                    template="plotly_dark",
                    height=380,
                    dragmode="pan",
                    margin=dict(l=10, r=10, t=20, b=10),
                    xaxis_title="Tarih",
                    yaxis_title="Portföy Değeri (TL, başlangıç = 100)",
                    xaxis=dict(
                        nticks=6,
                        tickformat="%d %b '%y",
                        tickangle=0
                    ),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                    hovermode="x unified"
                )
                st.info("Nasıl Okunmalı: Gri kesikli çizgi, hisseyi alıp hiç satmasaydınız paranızın durumunu; yeşil çizgi ise sadece model Yükseliş dediğinde hissede kalsaydınız paranızın durumunu gösterir. Yeşil çizginin gri çizginin üzerinde olması, modelin pasif beklemeye göre daha fazla kazandırdığını (veya daha az kaybettirdiğini) ifade eder.")
                st.plotly_chart(
                    fig_backtest,
                    use_container_width=True,
                    config={"displaylogo": False, "scrollZoom": True}
                )

                if len(backtest_df) > 0:
                    st.caption(
                        f"Test dönemi sonu: Pasif **{backtest_df['Pasif'].iloc[-1]:.2f} TL** | "
                        f"Model **{backtest_df['Model'].iloc[-1]:.2f} TL**. "
                        "İşlem maliyeti, komisyon ve kayma hesaba katılmamıştır; geçmiş performans gelecek getirinin garantisi değildir."
                    )

            # 3. SEKME: İstatistiksel Metrikler (Out-of-Sample test seti üzerinden)
            with sekme_istatistik:
                st.info("Araştırmacı Notu: Bu sekmedeki metrikler (Sınıflandırma Raporu ve Karmaşıklık Matrisi), modelin teknik davranışını, sınıf dengesizliklerini ve hata dağılımlarını analiz etmek içindir (Laboratuvar Testi). Günlük alım-satım kararlarında modelin gerçek dünya performansını görmek için lütfen Fiyat ve Tahmin sekmesindeki Walk-Forward CV (İleriye Yönelik Doğrulama) oranını referans alınız.")
                st.caption(f"Değerlendirme, modelin eğitimde hiç görmediği kronolojik son %20'lik test seti ({len(y_test)} gün) üzerinden yapılmıştır.")

                metrik_col1, metrik_col2 = st.columns(2)

                # --- A) Sınıflandırma Raporu (Precision / Recall / F1-Score) ---
                with metrik_col1:
                    st.subheader("📑 Sınıflandırma Raporu")
                    rapor_dict = classification_report(
                        y_test, y_pred,
                        labels=[0, 1],
                        target_names=["Düşüş", "Yükseliş"],
                        output_dict=True,
                        zero_division=0
                    )
                    rapor_df = pd.DataFrame(rapor_dict).T
                    rapor_df = rapor_df.rename(columns={
                        "precision": "Kesinlik (Precision)",
                        "recall": "Duyarlılık (Recall)",
                        "f1-score": "F1-Skoru",
                        "support": "Örnek Sayısı"
                    })
                    # 'accuracy' satırında sadece tek değer vardır; okunabilirlik için ayrı gösterilir
                    dogruluk = rapor_df.loc["accuracy", "F1-Skoru"] if "accuracy" in rapor_df.index else None
                    rapor_df = rapor_df.drop(index="accuracy", errors="ignore")
                    rapor_df["Örnek Sayısı"] = rapor_df["Örnek Sayısı"].astype(int)
                    st.dataframe(
                        rapor_df.style.format({
                            "Kesinlik (Precision)": "{:.2f}",
                            "Duyarlılık (Recall)": "{:.2f}",
                            "F1-Skoru": "{:.2f}"
                        }),
                        use_container_width=True
                    )
                    if dogruluk is not None:
                        st.caption(f"Test Seti Genel Doğruluk (Accuracy): **%{dogruluk*100:.2f}**")

                # --- B) Karmaşıklık Matrisi (Confusion Matrix) ---
                with metrik_col2:
                    st.subheader("🧩 Karmaşıklık Matrisi")
                    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
                    etiketler = ["Düşüş", "Yükseliş"]

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
                        yaxis=dict(autorange="reversed", fixedrange=True)  # Köşegen sol-üstten sağ-alta doğru görünsün
                    )
                    st.plotly_chart(
                        fig_cm,
                        use_container_width=True,
                        config=PLOTLY_STATIK_CONFIG
                    )

            # 4. SEKME: Haber Akışı (Google News RSS)
            with sekme_haber:
                temiz_hisse_adi = secilen_sembol.replace(".IS", "")
                st.subheader(f"📰 {temiz_hisse_adi} Güncel Haber Akışı")

                rss_url = f"https://news.google.com/rss/search?q={urllib.parse.quote(temiz_hisse_adi)}+hisse+haber&hl=tr&gl=TR&ceid=TR:tr"

                try:
                    istek = urllib.request.Request(
                        rss_url,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                    with urllib.request.urlopen(istek, timeout=10) as yanit:
                        xml_verisi = yanit.read()

                    kok = ET.fromstring(xml_verisi)
                    haber_ogeleri = kok.findall(".//item")[:5]
                except Exception:
                    haber_ogeleri = []

                if haber_ogeleri:
                    for item in haber_ogeleri:
                        baslik = item.findtext("title", default="Başlık Yok")
                        link = item.findtext("link", default="#")
                        yayin_tarihi = item.findtext("pubDate", default="Tarih Belirtilmemiş")
                        kaynak = item.findtext("source", default="Google Haberler")

                        with st.container(border=True):
                            st.markdown(f"#### [{baslik}]({link})")
                            st.caption(f"🏢 **Kaynak:** {kaynak} &nbsp;|&nbsp; 🕒 **Yayınlanma Tarihi:** {yayin_tarihi}")
                else:
                    st.warning("Bu hisse için güncel haber bulunamadı.")

        else:
            st.error("Veri çekme hatası oluştu.")

else:
    st.info(f"Lütfen bir hisse seçin ve 'Analizi Başlat' butonuna tıklayın. (Şu an seçili: {secilen_hisse_str})")

st.markdown("---")
st.caption("⚖️ **Yasal Uyarı:** Bu panel akademik bir istatistiksel modelleme projesi olup, üretilen tahminler kesinlikle yatırım tavsiyesi (YTD) değildir.")

st.sidebar.markdown("---")
st.sidebar.caption("Sürüm: 1.4 | XGBoost & TradingView Lightweight Charts")