# =================================================
# TITAN OMEGA v27.3 – ELEŞTİREL VE DENGELİ SKORLAMA
# =================================================
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import feedparser
import nltk
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# ====================== KÜRESEL PRESETLER ======================
COUNTRY_PRESETS = {
    "TR": {"rf": 35.0, "mrp": 7.0, "crp": 4.5, "g": 3.0, "tax": 25.0},
    "US": {"rf": 4.4, "mrp": 5.0, "crp": 0.0, "g": 2.2, "tax": 21.0},
    "DE": {"rf": 2.4, "mrp": 5.1, "crp": 0.0, "g": 1.7, "tax": 30.0},
    "GB": {"rf": 4.2, "mrp": 5.3, "crp": 0.0, "g": 2.0, "tax": 19.0},
}
SUFFIX_MAP = {"IS": "TR", "SA": "BR", "NS": "IN", "DE": "DE", "L": "GB", "TO": "CA"}
def detect_country(ticker):
    ticker = ticker.upper().split("^")[0].strip()
    if "." in ticker:
        suffix = ticker.split(".")[-1]
        return SUFFIX_MAP.get(suffix, "US")
    return "US"

# ====================== SİSTEM KURULUMU ve Önbellekleme ======================
@st.cache_resource
def get_vader():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except:
        nltk.download('vader_lexicon', quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    sia.lexicon.update({'beat':3.0,'missed':-3.0,'surge':3.0,'plunge':-3.0,'bullish':2.5,'bearish':-2.5})
    return sia

vader = get_vader()
st.set_page_config(page_title="TITAN OMEGA v27.3 – ELEŞTİREL", page_icon="🧿", layout="wide")

# ====================== TASARIM ======================
st.markdown("""
<style>
    .stApp {background: linear-gradient(135deg, #000000, #0a001a); color:#e0e0e0;}
    .big-title {font-size:80px !important; font-weight:900; text-align:center;
                 background:linear-gradient(90deg,#00ff88,#00ffff,#ff00ff,#ffff00);
                 -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                 text-shadow:0 0 40px rgba(0,255,136,0.6);}
</style>
""", unsafe_allow_html=True)
st.markdown("<h1 class='big-title'>TITAN OMEGA v27.3</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align:center; color:#00ffff;'>ELEŞTİREL SKORLAMA REVİZYONU</h2>", unsafe_allow_html=True)

# ====================== VERİ ÇEKME FONKSİYONLARI ======================
@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker) 
        info = stock.info
        hist = stock.history(period="2y")
        return hist, info
    except Exception as e:
        return pd.DataFrame(), {}

@st.cache_data(ttl=600, show_spinner=False)
def get_sentiment(ticker):
    try:
        feed = feedparser.parse(f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US')
        scores = []
        for entry in feed.entries[:10]:
            score = vader.polarity_scores(entry.title)['compound']
            scores.append(score)
        if not scores: return 0
        return np.mean(scores)
    except Exception as e:
        return 0 

# ====================== FİNANSAL MOTORLAR ======================
class ApexFundamentalEngine:
    @staticmethod
    def calculate_wacc(info, rf, mrp, crp, tax_rate, price):
        ie = info.get("interestExpense", 0) 
        total_debt = info.get("totalDebt", 0) 
        
        kd = rf + 0.025
        if total_debt > 0 and ie > 0:
            kd = (ie / total_debt)
        
        beta = info.get("beta", 1.0) or 1.0
        ke = rf + beta * (mrp + crp) 

        shares = info.get('sharesOutstanding') or info.get('impliedSharesOutstanding', 1e9)
        mcap = price * shares 
        
        cash = info.get("totalCash", 0)
        net_debt = max(total_debt - cash, 0) 
        ev = mcap + net_debt

        if ev <= 0 or mcap <= 0: return ke

        wacc = (mcap / ev * ke) + (net_debt / ev * kd * (1 - tax_rate))
        return wacc

    @staticmethod
    def calculate_dcf(info, wacc, shares, g_perpetual):
        fcf = info.get('freeCashflow', 0)
        revenue_growth = info.get('revenueGrowth', 0.05) or 0.05
        
        g = min(revenue_growth, 0.15)
        val = 0; curr = fcf
        
        for i in range(1, 6):
            curr *= (1 + g)
            val += curr / (1 + wacc)**i
            g *= 0.90 

        if wacc <= g_perpetual + 0.005: return 0
        
        term = (curr * (1 + g_perpetual)) / (wacc - g_perpetual)
        
        ev_calc = val + term / (1 + wacc)**5
        cash = info.get("totalCash", 0)
        total_debt = info.get("totalDebt", 0)
        equity_value = ev_calc - total_debt + cash
        
        dcf_price = equity_value / shares if shares > 0 else 0
        return dcf_price

    @staticmethod
    def calculate_valuation_metrics(info, price, rf, mrp, crp, tax_rate):
        shares = info.get('sharesOutstanding') or info.get('impliedSharesOutstanding', 1)
        
        country_code = detect_country(info.get('symbol', ''))
        preset = COUNTRY_PRESETS.get(country_code, COUNTRY_PRESETS["US"])
        g_perpetual = preset['g'] / 100
        
        wacc = ApexFundamentalEngine.calculate_wacc(info, rf, mrp, crp, tax_rate, price)
        dcf = ApexFundamentalEngine.calculate_dcf(info, wacc, shares, g_perpetual)
        
        eps, bvps = info.get('trailingEps', 0), info.get('bookValue', 0)
        revenue_growth = info.get('revenueGrowth', 0.05) * 100
        
        graham = np.sqrt(22.5 * eps * bvps) if eps > 0 and bvps > 0 else 0
        lynch = eps * revenue_growth if eps > 0 and revenue_growth > 0 else 0
        
        vals = [v for v in [dcf, graham, lynch] if v > 0]
        if not vals: return {"error":"Veri eksik", "wacc":wacc}
        
        fair = sum(vals) / len(vals)
        upside = (fair - price) / price * 100
        
        net_income = info.get('netIncomeToCommon', 0)
        revenue = info.get('totalRevenue', 1)
        
        net_margin = (net_income / revenue) if revenue > 0 and revenue != 1 else 0
        debt_to_equity = info.get('debtToEquity', 1000)
        p_e = info.get('trailingPE', 99)
        
        return {
            "fair_price": fair, "upside": upside, "wacc": wacc, "error": None,
            "net_margin": net_margin, "debt_to_equity": debt_to_equity, "p_e": p_e,
        }

# ====================== TEKNİK MOTOR (MACD EKLENDİ) ======================

class ApexTechnicalEngine:
    
    @staticmethod
    def calculate_macd(df, fast=12, slow=26, signal=9):
        # MACD Hesaplaması
        ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        # Sinyal Kontrolü
        macd_sig = "NÖTR"
        if macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]:
            macd_sig = "MACD AL SİNYALİ"
        elif macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]:
            macd_sig = "MACD SAT SİNYALİ"
            
        return macd_sig, macd_line.iloc[-1], signal_line.iloc[-1]


    @staticmethod
    def calculate_adx(df, period=14):
        # ADX Hesaplaması (Değişmedi)
        plus_dm = df['High'].diff(); minus_dm = df['Low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
        tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(),
                        (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        return dx.ewm(alpha=1/period, adjust=False).mean().iloc[-1]

    @staticmethod
    def analyze(df):
        if df.empty: return None
        c, h, l = df['Close'], df['High'], df['Low']
        delta = c.diff(); gain = delta.where(delta>0,0).ewm(alpha=1/14).mean()
        loss = -delta.where(delta<0,0).ewm(alpha=1/14).mean()
        rsi = 100 - (100/(1 + gain/loss))
        sma20 = c.rolling(20).mean(); std20 = c.rolling(20).std()
        upper, lower = sma20 + 2*std20, sma20 - 2*std20
        
        # MACD Entegrasyonu
        macd_sig, macd_val, signal_val = ApexTechnicalEngine.calculate_macd(df)

        # Trend Kesişimleri (SMA 50/200)
        sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
        cross = "NÖTR"
        if sma50.iloc[-1] > sma200.iloc[-1] and sma50.iloc[-2] <= sma200.iloc[-2]: cross = "GOLDEN CROSS (RALLİ)"
        elif sma50.iloc[-1] < sma200.iloc[-1] and sma50.iloc[-2] >= sma200.iloc[-2]: cross = "DEATH CROSS (ÇÖKÜŞ)"
        elif sma50.iloc[-1] > sma200.iloc[-1]: cross = "BULL TREND"
        else: cross = "BEAR TREND"
        
        # ADX ve Ichimoku
        adx = ApexTechnicalEngine.calculate_adx(df)
        strength = "GÜÇLÜ" if adx > 25 else "ZAYIF" if adx < 20 else "NORMAL"
        tenkan = (h.rolling(9).max() + l.rolling(9).min())/2
        kijun = (h.rolling(26).max() + l.rolling(26).min())/2
        ichi_sig = "BULLISH" if tenkan.iloc[-1] > kijun.iloc[-1] else "BEARISH"
        
        # Fibonacci (Değişmedi)
        fib_max, fib_min = h.iloc[-252:].max(), l.iloc[-252:].min()
        diff = fib_max - fib_min
        fib = {"0%":fib_max, "23.6%":fib_max-0.236*diff, "38.2%":fib_max-0.382*diff,
               "50%":fib_max-0.5*diff, "61.8%":fib_max-0.618*diff, "100%":fib_min}
               
        return {"rsi":rsi.iloc[-1], "upper":upper, "lower":lower, "cross":cross, 
                "adx":adx, "strength":strength, "ichimoku":ichi_sig, "fib":fib, 
                "macd_sig":macd_sig, "macd_val":macd_val, "signal_val":signal_val}


# ====================== ANA UYGULAMA MANTIĞI ======================
def main():
    # Kenar Çubuğu Parametreleri
    with st.sidebar:
        st.header("TITAN KONTROL")
        ticker = st.text_input("Hisse Kodu", "THYAO.IS").upper().strip()
        
        country_code = detect_country(ticker)
        preset = COUNTRY_PRESETS.get(country_code, COUNTRY_PRESETS["US"])
        st.info(f"📍 Otomatik Ülke: {country_code} ({COUNTRY_PRESETS.get(country_code, COUNTRY_PRESETS['US'])['tax']}% Vergi)")
        
        # Finansal Ayarlar (Float Dönüşümleri yapıldı)
        rf = st.slider("Risk Free Rate %", 0.0, 100.0, preset["rf"], 0.1) / 100
        mrp = st.slider("Piyasa Risk Primi %", 3.0, 15.0, preset["mrp"], 0.1) / 100
        crp = st.slider("Ülke Risk Primi %", 0.0, 25.0, preset["crp"], 0.1) / 100
        tax_rate = st.slider("Vergi Oranı %", 0.0, 50.0, preset["tax"], 1.0) / 100


    if st.button("APEX ANALİZİ BAŞLAT (v27.3)", type="primary", use_container_width=True):
        with st.spinner("TITAN v27.3 Veri Topluyor ve Eleştirel Analiz Ediyor..."):
            
            hist, info = get_stock_data(ticker) 
            
            if hist.empty or not info or not info.get('symbol'):
                st.error("Hisse bulunamadı, veri çekilemedi veya sembol bilgisi eksik!"); return
            
            price = hist['Close'].iloc[-1]
            tech = ApexTechnicalEngine.analyze(hist)
            fund = ApexFundamentalEngine.calculate_valuation_metrics(info, price, rf, mrp, crp, tax_rate)
            sentiment_score = get_sentiment(ticker) 

            # ====================== SKORLAMA MANTIĞI (ELEŞTİREL - V27.3) ======================
            score = 50 

            # A. Teknik Skorlama (Eleştirel Dengeleme)
            if "GOLDEN" in tech['cross']: score += 10 # Önceki: 15
            elif "BULL TREND" in tech['cross']: score += 8
            elif "DEATH" in tech['cross']: score -= 25 # CEZA ARTIRILDI: -15 idi
            
            if tech['strength'] == "GÜÇLÜ": score += 8
            if tech['rsi'] < 30: score += 10 
            
            # MACD Skoru (Eleştirel Dengeleme)
            if "MACD AL" in tech['macd_sig']: score += 8 # Önceki: 10
            elif "MACD SAT" in tech['macd_sig']: score -= 15 # CEZA ARTIRILDI: -10 idi
            
            # B. Temel Skorlama (Eleştirel Dengeleme)
            if not fund.get("error"):
                # Adil Değer Potansiyeli
                if fund['upside'] > 30: score += 25 # Önceki: 30
                elif fund['upside'] > 10: score += 10 # Önceki: 15
                elif fund['upside'] < -15: score -= 30 # CEZA ARTIRILDI: -20 idi (En büyük ceza)

                # Kalite Metrikleri (Dengeli tutuldu)
                if fund['net_margin'] > 0.05: score += 8
                if fund['debt_to_equity'] < 100: score += 8
                if fund['p_e'] < 20 and fund['p_e'] > 0: score += 8

            # C. Duygu (Sentiment) Skorlama 
            sentiment_impact = sentiment_score * 15 
            score += sentiment_impact

            # Puanı 0 ile 100 arasında sınırla
            score = max(min(score, 100), 0)

            # ====================== GÖRSELLEŞTİRME ======================
            st.success(f"ANALİZ TAMAMLANDI → {info.get('longName', ticker)}")
            c1, c2, c3, c4, c5 = st.columns(5)
            
            signal = "GÜÇLÜ AL" if score>=85 else "AL" if score>=65 else "TUT" if score>=40 else "SAT"
            
            c1.metric("Fiyat", f"{price:,.2f} {info.get('currency','USD')}")
            c2.metric("TITAN SKORU (v27.3)", f"{int(score)}/100")
            c3.metric("SİNYAL", signal)
            c4.metric("Adil Değer", f"{fund.get('fair_price',0):,.2f}" if not fund.get("error") else "N/A")
            c5.metric("Duygu (Sentiment)", f"{sentiment_score:+.2f}")

            st.markdown("---")
            
            # Detay Tablosu
            st.subheader("⚙️ REVİZYON DETAYLARI")
            col_rev1, col_rev2 = st.columns(2)
            
            with col_rev1:
                st.markdown("#### Temel & Değerleme")
                st.table({
                    "Metrik": ["WACC", "Adil Fiyat Potansiyeli", "Net Kar Marjı", "Borç/Özkaynak"],
                    "Değer": [
                        f"%{fund['wacc']*100:.2f}" if fund.get("wacc") else "N/A",
                        f"%{fund.get('upside',0):+.1f}" if not fund.get("error") else "N/A",
                        f"%{fund.get('net_margin',0)*100:.1f}",
                        f"{fund.get('debt_to_equity',0):.0f}"
                    ]
                })

            with col_rev2:
                st.markdown("#### Teknik & Trend")
                st.table({
                    "Metrik": ["Ana Trend (SMA)", "MACD Sinyali", "Trend Gücü (ADX)", "RSI"],
                    "Değer": [
                        tech['cross'],
                        tech['macd_sig'],
                        f"{tech['adx']:.1f} ({tech['strength']})",
                        f"{tech['rsi']:.1f}",
                    ]
                }) # <--- Hata burada düzeltildi.

            # Grafikler
            col1, col2 = st.columns([2,1])
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=hist.index[-150:], open=hist['Open'][-150:], high=hist['High'][-150:], 
                                             low=hist['Low'][-150:], close=hist['Close'][-150:], name="Fiyat"))
                fig.add_trace(go.Scatter(x=hist.index[-150:], y=tech['upper'][-150:], name="BB Üst", line=dict(color="#00ffff")))
                fig.add_trace(go.Scatter(x=hist.index[-150:], y=tech['lower'][-150:], name="BB Alt", line=dict(color="#ff00ff")))
                for k,v in tech['fib'].items():
                    fig.add_hline(y=v, line_dash="dot", annotation_text=f"Fib {k}") 
                fig.update_layout(template="plotly_dark", height=600, title="Fiyat Hareketi, Bollinger ve Fibonacci Seviyeleri")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown(f"#### 📰 PİYASA DUYGUSU")
                st.markdown(f"**Son Haber Duygusu (Ortalama):** <span style='color: {'#00ff88' if sentiment_score > 0.1 else '#ff3366' if sentiment_score < -0.1 else '#ffaa00'}'>{sentiment_score:+.2f}</span>", unsafe_allow_html=True)
                st.markdown(f"**Etki:** {sentiment_impact:+.1f} Puan")
                st.markdown("---")
                st.markdown(f"#### 📈 MACD MOMENTUM")
                st.markdown(f"**MACD Hattı:** {tech['macd_val']:.3f}")
                st.markdown(f"**Sinyal Hattı:** {tech['signal_val']:.3f}")

            if score >= 85: st.balloons()
            st.markdown("---")


if __name__ == "__main__":
    main()
    st.caption("© TITAN OMEGA v27.3 – Eleştirel Skorlama | 2025 | Yatırım tavsiyesi değildir")
