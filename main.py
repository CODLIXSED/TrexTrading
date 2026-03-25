import os
import asyncio
import requests
import pandas as pd
import numpy as np
import time
import joblib
from datetime import datetime, timedelta
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sklearn.linear_model import SGDClassifier
import yfinance as yf

# ================= CONFIG =================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

STOCKS = ['BBCA','BBRI','BMRI','TLKM','ASII','ADRO','ANTM','UNVR','ICBP','INDF']  # bisa tambah

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ================= DATA ENGINE =================
class DataEngine:
    def __init__(self):
        self.cache = {}
        self.cache_time = {}

    async def get(self, symbol):
        if symbol in self.cache and time.time() - self.cache_time[symbol] < 60:
            return self.cache[symbol]

        sources = [self.stockbit, self.yahoo]

        for s in sources:
            try:
                d = await s(symbol)
                if d and d['price'] > 0:
                    self.cache[symbol] = d
                    self.cache_time[symbol] = time.time()
                    return d
            except:
                continue
        return None

    async def stockbit(self, symbol):
        try:
            url = f"https://finstock-api.stockbit.com/stock/{symbol}"
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                d = r.json()
                return {
                    "price": float(d['last_price']),
                    "change": float(d.get('change_pct', 0)),
                    "volume": int(d.get('volume', 0))
                }
        except:
            return None

    async def yahoo(self, symbol):
        try:
            t = yf.Ticker(f"{symbol}.JK")
            h = t.history(period="1d")
            return {
                "price": float(h['Close'].iloc[-1]),
                "change": 0,
                "volume": int(h['Volume'].iloc[-1])
            }
        except:
            return None

data_engine = DataEngine()

# ================= ML =================
def get_hist(symbol):
    try:
        df = yf.Ticker(f"{symbol}.JK").history(period="6mo")
        return df.dropna()
    except:
        return None

def features(df):
    df['ret'] = df['Close'].pct_change()
    df['ma5'] = df['Close'].rolling(5).mean()
    df['ma20'] = df['Close'].rolling(20).mean()
    df['vol'] = df['Volume'].rolling(5).mean()
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()
    return df[['ret','ma5','ma20','vol']], df['target']

def load_model(s):
    path = f"{MODEL_DIR}/{s}.pkl"
    return joblib.load(path) if os.path.exists(path) else None

def save_model(s, m):
    joblib.dump(m, f"{MODEL_DIR}/{s}.pkl")

def update_model(s):
    df = get_hist(s)
    if df is None or len(df) < 50:
        return None

    X,y = features(df)
    m = load_model(s)

    if m is None:
        m = SGDClassifier(loss='log_loss')
        m.fit(X,y)
    else:
        m.partial_fit(X,y)

    save_model(s,m)
    return m

def predict(s):
    df = get_hist(s)
    if df is None:
        return None
    m = update_model(s)
    X,_ = features(df)
    return m.predict_proba(X.iloc[-1:])[0][1]*100

# ================= SIGNAL =================
async def analyze(s):
    q = await data_engine.get(s)
    if not q: return None

    if q['price'] < 100 or q['volume'] < 500000:
        return None

    prob = predict(s)
    if prob is None or prob < 65:
        return None

    entry = round(q['price'], -2)
    target = round(q['price']*1.06, -2)
    sl = round(q['price']*0.975, -2)

    return {
        "symbol": s,
        "entry": entry,
        "target": target,
        "sl": sl,
        "prob": round(prob,1)
    }

# ================= SCAN =================
async def scan():
    sem = asyncio.Semaphore(5)

    async def task(s):
        async with sem:
            return await analyze(s)

    res = await asyncio.gather(*[task(s) for s in STOCKS])
    sig = [r for r in res if r]
    return sorted(sig, key=lambda x: x['prob'], reverse=True)[:5]

# ================= TELEGRAM =================
bot = Bot(token=TELEGRAM_TOKEN)
app = Application.builder().token(TELEGRAM_TOKEN).build()

async def start(update: Update, context):
    await update.message.reply_text("🚀 AI TRADING BOT LIVE\n/fullscan")

async def fullscan(update: Update, context):
    await update.message.reply_text("Scanning...")
    s = await scan()
    msg = "🧠 SIGNAL:\n\n"
    for i,x in enumerate(s,1):
        msg += f"{i}. {x['symbol']}\nProb: {x['prob']}%\nEntry:{x['entry']}\nTarget:{x['target']}\nSL:{x['sl']}\n\n"
    await update.message.reply_text(msg)

app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("fullscan", fullscan))

# ================= AUTO =================
scheduler = AsyncIOScheduler()

async def auto():
    s = await scan()
    msg = "🔥 AUTO SIGNAL\n\n"
    for x in s:
        msg += f"{x['symbol']} ({x['prob']}%)\n"
    await bot.send_message(chat_id=CHAT_ID, text=msg)

scheduler.add_job(auto, 'cron', hour=8)
scheduler.add_job(auto, 'cron', hour=15)

# ================= RUN =================
async def main():
    scheduler.start()
    await app.run_polling()

asyncio.run(main())
