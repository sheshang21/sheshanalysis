from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import asyncio
import json
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from scipy import stats
import warnings
import traceback

warnings.filterwarnings('ignore')

app = FastAPI(title="NSE Stock Analyzer API")

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active WebSocket connections
active_connections: List[WebSocket] = []

# NSE Stock Universe
NSE_STOCKS = [
    'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 'ITC', 'SBIN',
    'BHARTIARTL', 'KOTAKBANK', 'LT', 'AXISBANK', 'ASIANPAINT', 'MARUTI', 'HCLTECH',
    'BAJFINANCE', 'WIPRO', 'SUNPHARMA', 'TITAN', 'ULTRACEMCO', 'NESTLEIND', 'ONGC',
    'TATAMOTORS', 'NTPC', 'POWERGRID', 'JSWSTEEL', 'M&M', 'TECHM', 'ADANIENT', 'ADANIPORTS',
    'COALINDIA', 'HINDALCO', 'TATASTEEL', 'BAJAJFINSV', 'DIVISLAB', 'DRREDDY', 'GRASIM',
    'CIPLA', 'BRITANNIA', 'EICHERMOT', 'HEROMOTOCO', 'APOLLOHOSP', 'INDUSINDBK', 'UPL',
    'BPCL', 'SBILIFE', 'HDFCLIFE', 'BAJAJ-AUTO', 'VEDL', 'TATACONSUM', 'DIXON', 'POLYCAB',
    'PERSISTENT', 'COFORGE', 'LTIM', 'ZOMATO', 'PAYTM', 'NAUKRI', 'IRCTC', 'DMART',
    'TRENT', 'BEL', 'HAL', 'ADANIGREEN', 'GODREJCP', 'DABUR', 'PIDILITIND', 'HAVELLS'
]

SECTOR_MAP = {
    'RELIANCE': 'Energy', 'TCS': 'IT', 'HDFCBANK': 'Banking', 'INFY': 'IT', 'ICICIBANK': 'Banking',
    'HINDUNILVR': 'FMCG', 'ITC': 'FMCG', 'SBIN': 'Banking', 'BHARTIARTL': 'Telecom', 'KOTAKBANK': 'Banking',
    'LT': 'Infrastructure', 'AXISBANK': 'Banking', 'ASIANPAINT': 'Paints', 'MARUTI': 'Auto', 'HCLTECH': 'IT',
    'DIXON': 'Electronics', 'POLYCAB': 'Cables', 'PERSISTENT': 'IT', 'COFORGE': 'IT', 'LTIM': 'IT',
    'ZOMATO': 'Tech', 'PAYTM': 'FinTech', 'NAUKRI': 'Tech', 'IRCTC': 'Travel', 'DMART': 'Retail'
}

# Cache for price data
price_cache: Dict[str, Dict] = {}
last_fetch_time: Dict[str, datetime] = {}

# ==================== UTILITY FUNCTIONS ====================

def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    if len(prices) < 26:
        return 0
    ema12 = calculate_ema(prices, 12)
    ema26 = calculate_ema(prices, 26)
    return ema12 - ema26

def calculate_ema(prices, period):
    multiplier = 2 / (period + 1)
    ema = np.mean(prices[:period])
    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema
    return ema

def calculate_bb_position(prices, period=20):
    if len(prices) < period:
        return 50
    recent = prices[-period:]
    sma = np.mean(recent)
    std = np.std(recent)
    upper = sma + (2 * std)
    lower = sma - (2 * std)
    current = prices[-1]
    if upper == lower:
        return 50
    return max(0, min(100, ((current - lower) / (upper - lower)) * 100))

def calculate_volume_multiple(volumes):
    if len(volumes) < 20:
        return 1.0
    current = volumes[-1]
    avg20 = np.mean(volumes[-20:])
    return current / avg20 if avg20 > 0 else 1.0

def detect_trend(prices):
    if len(prices) < 5:
        return 'Sideways'
    recent = prices[-5:]
    ups = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i-1])
    if ups >= 4:
        return 'Strong Uptrend'
    elif ups >= 3:
        return 'Uptrend'
    elif ups <= 1:
        return 'Downtrend'
    return 'Sideways'

def detect_patterns(closes, highs, lows, volumes, dates):
    """Detect chart patterns"""
    patterns = []
    
    # Cup and Handle
    if len(closes) >= 60:
        lookback = 60
        recent = closes[-lookback:]
        cup_depth = (np.max(recent[:30]) - np.min(recent[15:45])) / np.max(recent[:30])
        if 0.12 <= cup_depth <= 0.33:
            patterns.append({'name': '☕ Cup & Handle', 'confidence': 75})
    
    # 52-Week Breakout
    if len(closes) >= 52:
        current = closes[-1]
        year_high = np.max(highs[-52:-5])
        if current >= year_high * 0.995:
            patterns.append({'name': '🚀 52W Breakout', 'confidence': 85})
    
    # VCP Pattern
    if len(closes) >= 90:
        segments = np.array_split(closes[-90:], 4)
        contractions = [(seg.max() - seg.min()) / seg.max() * 100 for seg in segments]
        if len(contractions) >= 3 and contractions[-1] < contractions[-2] < contractions[-3]:
            patterns.append({'name': '📉 VCP', 'confidence': 80})
    
    return patterns

# ==================== API ENDPOINTS ====================

@app.get("/")
async def read_root():
    """Serve the main HTML page"""
    try:
        with open("index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return {"message": "Frontend not found. Deploy index.html in same directory."}

@app.get("/api/stocks")
async def get_stocks():
    """Get list of available NSE stocks"""
    return {
        "stocks": NSE_STOCKS,
        "sectors": SECTOR_MAP,
        "total": len(NSE_STOCKS)
    }

@app.get("/api/price/{symbol}")
async def get_live_price(symbol: str):
    """Get live price for a stock - used by WebSocket"""
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        data = ticker.history(period="1d", interval="1m")
        
        if data.empty:
            raise HTTPException(status_code=404, detail="No data found")
        
        current_price = float(data['Close'].iloc[-1])
        prev_close = float(data['Close'].iloc[0])
        change = ((current_price - prev_close) / prev_close) * 100
        volume = float(data['Volume'].iloc[-1])
        
        return {
            "symbol": symbol,
            "price": round(current_price, 2),
            "change": round(change, 2),
            "volume": int(volume),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/scan")
async def scan_stocks(request: dict):
    """Scan multiple stocks with analysis"""
    try:
        symbols = request.get("symbols", NSE_STOCKS[:50])
        mode = request.get("mode", "quick")  # quick, pattern, beta
        
        results = []
        for symbol in symbols:
            try:
                # Fetch historical data
                ticker = yf.Ticker(f"{symbol}.NS")
                hist = ticker.history(period="3mo", interval="1d")
                
                if hist.empty or len(hist) < 30:
                    continue
                
                closes = hist['Close'].values
                highs = hist['High'].values
                lows = hist['Low'].values
                volumes = hist['Volume'].values
                
                price = closes[-1]
                prev_close = closes[-2] if len(closes) > 1 else price
                change = ((price - prev_close) / prev_close) * 100
                
                # Technical indicators
                rsi = calculate_rsi(closes)
                macd = calculate_macd(closes)
                bb = calculate_bb_position(closes)
                vol = calculate_volume_multiple(volumes)
                trend = detect_trend(closes)
                
                # Pattern detection for pattern mode
                patterns = []
                if mode == "pattern":
                    patterns = detect_patterns(closes, highs, lows, volumes, hist.index)
                
                # Scoring
                score = 0
                if mode == "pattern":
                    score += min(35, len(patterns) * 12)
                
                # RSI scoring
                if 58 <= rsi <= 65:
                    score += 25
                elif 52 <= rsi <= 68:
                    score += 18
                elif 35 <= rsi <= 42:
                    score += 20
                
                # MACD scoring
                if macd > 10:
                    score += 20
                elif macd > 5:
                    score += 15
                elif macd > 0:
                    score += 10
                
                # Volume scoring
                if vol >= 3.0:
                    score += 20
                elif vol >= 2.0:
                    score += 15
                elif vol >= 1.5:
                    score += 10
                
                # Trend scoring
                if trend == 'Strong Uptrend':
                    score += 15
                elif trend == 'Uptrend':
                    score += 10
                
                # Daily change
                if change >= 5:
                    score += 10
                elif change >= 3:
                    score += 8
                elif change >= 2:
                    score += 5
                
                # Rating
                if score >= 90:
                    rating = 'Excellent'
                elif score >= 80:
                    rating = 'Very Good'
                elif score >= 70:
                    rating = 'Good'
                elif score >= 65:
                    rating = 'Fair'
                else:
                    rating = 'Watchlist'
                
                results.append({
                    'symbol': symbol,
                    'price': round(price, 2),
                    'change': round(change, 2),
                    'rsi': round(rsi, 1),
                    'macd': round(macd, 2),
                    'bb': round(bb, 1),
                    'volume_multiple': round(vol, 2),
                    'trend': trend,
                    'score': score,
                    'rating': rating,
                    'patterns': patterns,
                    'sector': SECTOR_MAP.get(symbol, 'Other')
                })
                
            except Exception as e:
                print(f"Error scanning {symbol}: {e}")
                continue
        
        return {
            "results": results,
            "total_scanned": len(results),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze")
async def analyze_stock(request: dict):
    """Deep analysis of a single stock including beta"""
    try:
        symbol = request.get("symbol")
        period = request.get("period", "1y")
        
        # Fetch stock and Nifty data
        ticker = yf.Ticker(f"{symbol}.NS")
        nifty = yf.Ticker("^NSEI")
        
        # Get historical data
        end = datetime.now()
        if period == "1y":
            start = end - timedelta(days=365)
        elif period == "3y":
            start = end - timedelta(days=1095)
        else:
            start = end - timedelta(days=1825)
        
        stock_data = ticker.history(start=start, end=end)
        nifty_data = nifty.history(start=start, end=end)
        
        if stock_data.empty or nifty_data.empty:
            raise HTTPException(status_code=404, detail="No data found")
        
        # Calculate returns
        stock_returns = stock_data['Close'].pct_change() * 100
        nifty_returns = nifty_data['Close'].pct_change() * 100
        
        # Align data
        returns_df = pd.DataFrame({
            'Stock': stock_returns,
            'Nifty': nifty_returns
        }).dropna()
        
        if len(returns_df) < 30:
            raise HTTPException(status_code=400, detail="Insufficient data")
        
        # Calculate beta using regression
        X = add_constant(returns_df['Nifty'])
        model = OLS(returns_df['Stock'], X).fit()
        
        beta = float(model.params['Nifty'])
        alpha = float(model.params['const'])
        r_squared = float(model.rsquared)
        
        # Risk metrics
        volatility = float(returns_df['Stock'].std())
        annual_vol = volatility * np.sqrt(252)
        mean_return = float(returns_df['Stock'].mean())
        annual_return = mean_return * 252
        
        # Sharpe ratio (assuming 6.5% risk-free rate)
        rf_daily = 6.5 / 252
        sharpe = (mean_return - rf_daily) / volatility
        annual_sharpe = sharpe * np.sqrt(252)
        
        # Drawdown
        cum_returns = (1 + returns_df['Stock'] / 100).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max * 100
        max_drawdown = float(drawdown.min())
        
        # VaR
        var_95 = float(np.percentile(returns_df['Stock'], 5))
        
        # Win rate
        winning_days = (returns_df['Stock'] > 0).sum()
        win_rate = (winning_days / len(returns_df)) * 100
        
        # Current price metrics
        current_price = float(stock_data['Close'].iloc[-1])
        high_52w = float(stock_data['High'].iloc[-252:].max()) if len(stock_data) >= 252 else float(stock_data['High'].max())
        low_52w = float(stock_data['Low'].iloc[-252:].min()) if len(stock_data) >= 252 else float(stock_data['Low'].min())
        
        # Generate recommendation
        score = 0
        reasons = []
        
        if 0.8 <= beta <= 1.2:
            score += 1
            reasons.append("✅ Moderate beta (~1.0)")
        elif beta > 1.5:
            score -= 1
            reasons.append("⚠️ High volatility (beta >1.5)")
        
        if annual_sharpe > 1.5:
            score += 2
            reasons.append("✅ Excellent Sharpe ratio")
        elif annual_sharpe > 1.0:
            score += 1
            reasons.append("✅ Good Sharpe ratio")
        
        if max_drawdown > -15:
            score += 1
            reasons.append("✅ Low drawdown")
        elif max_drawdown < -30:
            score -= 2
            reasons.append("⚠️ High drawdown")
        
        if win_rate > 55:
            score += 1
            reasons.append("✅ High win rate")
        
        if score >= 4:
            recommendation = "BUY"
        elif score >= 1:
            recommendation = "HOLD"
        else:
            recommendation = "AVOID"
        
        return {
            "symbol": symbol,
            "price": round(current_price, 2),
            "high_52w": round(high_52w, 2),
            "low_52w": round(low_52w, 2),
            "beta": round(beta, 4),
            "alpha": round(alpha, 4),
            "r_squared": round(r_squared, 4),
            "annual_return": round(annual_return, 2),
            "annual_volatility": round(annual_vol, 2),
            "sharpe_ratio": round(annual_sharpe, 4),
            "max_drawdown": round(max_drawdown, 2),
            "var_95": round(var_95, 2),
            "win_rate": round(win_rate, 1),
            "recommendation": recommendation,
            "score": score,
            "reasons": reasons,
            "data_points": len(returns_df),
            "period": period
        }
        
    except Exception as e:
        print(f"Error analyzing {symbol}: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== WEBSOCKET FOR LIVE PRICES ====================

@app.websocket("/ws/prices")
async def websocket_prices(websocket: WebSocket):
    """WebSocket endpoint for continuous live price updates"""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        # Receive initial symbols list
        data = await websocket.receive_text()
        symbols = json.loads(data).get("symbols", ["RELIANCE", "TCS", "INFY"])
        
        print(f"WebSocket connected, tracking: {symbols}")
        
        # Continuous update loop
        while True:
            try:
                price_updates = []
                
                for symbol in symbols:
                    try:
                        # Check cache first (avoid rate limits)
                        now = datetime.now()
                        if symbol in last_fetch_time:
                            time_diff = (now - last_fetch_time[symbol]).seconds
                            if time_diff < 5 and symbol in price_cache:
                                # Use cached data if less than 5 seconds old
                                price_updates.append(price_cache[symbol])
                                continue
                        
                        # Fetch fresh data
                        ticker = yf.Ticker(f"{symbol}.NS")
                        
                        # Try intraday first
                        data = ticker.history(period="1d", interval="1m")
                        
                        if not data.empty and len(data) > 0:
                            current_price = float(data['Close'].iloc[-1])
                            prev_close = float(data['Close'].iloc[0])
                            change = ((current_price - prev_close) / prev_close) * 100
                            volume = int(data['Volume'].sum())
                            
                            price_info = {
                                "symbol": symbol,
                                "price": round(current_price, 2),
                                "change": round(change, 2),
                                "volume": volume,
                                "timestamp": now.strftime("%H:%M:%S"),
                                "status": "live"
                            }
                        else:
                            # Fallback to last close if intraday not available
                            data = ticker.history(period="5d")
                            if not data.empty:
                                current_price = float(data['Close'].iloc[-1])
                                prev_close = float(data['Close'].iloc[-2]) if len(data) > 1 else current_price
                                change = ((current_price - prev_close) / prev_close) * 100
                                
                                price_info = {
                                    "symbol": symbol,
                                    "price": round(current_price, 2),
                                    "change": round(change, 2),
                                    "volume": 0,
                                    "timestamp": now.strftime("%H:%M:%S"),
                                    "status": "delayed"
                                }
                            else:
                                continue
                        
                        # Update cache
                        price_cache[symbol] = price_info
                        last_fetch_time[symbol] = now
                        price_updates.append(price_info)
                        
                        # Small delay between stocks to avoid rate limits
                        await asyncio.sleep(0.2)
                        
                    except Exception as e:
                        print(f"Error fetching {symbol}: {e}")
                        # Send cached data if available
                        if symbol in price_cache:
                            price_updates.append(price_cache[symbol])
                
                # Send updates to client
                if price_updates:
                    await websocket.send_json({
                        "type": "price_update",
                        "data": price_updates,
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Wait before next update cycle (5-10 seconds)
                await asyncio.sleep(8)
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"WebSocket error: {e}")
                await asyncio.sleep(5)
                
    except Exception as e:
        print(f"WebSocket connection error: {e}")
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)
        print("WebSocket disconnected")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_connections": len(active_connections),
        "cached_stocks": len(price_cache),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)