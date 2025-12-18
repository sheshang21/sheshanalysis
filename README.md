# 📊 NSE Stock Analyzer - Live Real-Time Platform

A comprehensive stock analysis platform combining **live price updates**, **pattern detection**, and **beta analysis** in a single application. Built with FastAPI backend and vanilla JavaScript frontend for optimal performance.

## ✨ Features

### 🔴 Live Price Monitor
- **Continuous real-time updates** (every 8 seconds) via WebSocket
- No refresh button needed - truly live like broker sites
- Customizable watchlist
- Auto-reconnection on connection loss
- Visual pulse animations on updates

### 🔍 Stock Scanner
- Scans 50-200+ NSE stocks
- Technical indicators: RSI, MACD, Bollinger Bands, Volume
- Pattern detection: Cup & Handle, VCP, 52W Breakout
- Intelligent scoring system (0-100)
- One-click deep analysis from results

### 📈 Beta & Risk Analyzer
- Complete beta calculation vs NIFTY 50
- 30+ financial metrics
- Risk analysis (VaR, Sharpe, Sortino)
- BUY/HOLD/AVOID recommendations
- 1Y/3Y/5Y analysis periods

## 🚀 Quick Start (Local Development)

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# 1. Clone or create project directory
mkdir stock-analyzer
cd stock-analyzer

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the server
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# 5. Open browser
# Navigate to: http://localhost:8000
```

## 🌐 Vercel Deployment (Production)

### Method 1: GitHub + Vercel (Recommended)

1. **Create GitHub Repository**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/stock-analyzer.git
   git push -u origin main
   ```

2. **Deploy on Vercel**
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Import your GitHub repository
   - Vercel auto-detects Python and deploys
   - Your app will be live at: `https://your-project.vercel.app`

### Method 2: Vercel CLI

```bash
# Install Vercel CLI
npm i -g vercel

# Login to Vercel
vercel login

# Deploy
vercel

# Follow prompts, then:
vercel --prod
```

### ⚠️ Important Vercel Considerations

**WebSocket Limitations:**
- Vercel serverless functions have a **10-second timeout**
- WebSocket connections work but may disconnect after 10s
- For production, consider these alternatives:

#### Option A: Polling Mode (Recommended for Vercel)
Add this to `index.html` after line 380:

```javascript
// Replace WebSocket with polling for Vercel
let pollingInterval = null;

function startPolling() {
    updateStatus('Polling mode - Updates every 10s', true);
    
    pollingInterval = setInterval(async () => {
        const watchlist = document.getElementById('watchlist').value.split(',').map(s => s.trim());
        const priceData = [];
        
        for (const symbol of watchlist) {
            try {
                const response = await fetch(`${getAPIURL()}/api/price/${symbol}`);
                const data = await response.json();
                priceData.push(data);
            } catch (error) {
                console.error(`Error fetching ${symbol}:`, error);
            }
        }
        
        updatePrices(priceData);
        document.getElementById('lastUpdate').textContent = 
            `Last update: ${new Date().toLocaleTimeString()}`;
    }, 10000); // Poll every 10 seconds
}

// Replace connectWebSocket() call with:
window.addEventListener('load', () => {
    if (window.location.host.includes('vercel.app')) {
        startPolling();
    } else {
        connectWebSocket();
    }
});
```

#### Option B: Railway/Render (Better for WebSockets)
For true real-time WebSocket support, deploy to:
- **Railway.app** (free tier available)
- **Render.com** (free tier available)
- **Fly.io** (free tier available)

These platforms support long-running WebSocket connections.

## 📁 Project Structure

```
stock-analyzer/
├── app.py              # FastAPI backend (all API endpoints + WebSocket)
├── index.html          # Frontend (single page app)
├── requirements.txt    # Python dependencies
├── vercel.json        # Vercel deployment config
└── README.md          # This file
```

## 🔧 Configuration

### Environment Variables (Optional)
Create `.env` file:

```env
# API Configuration
API_RATE_LIMIT=100
CACHE_TTL=300

# CORS Origins (comma-separated)
ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com
```

### Customizing Stock List
Edit `NSE_STOCKS` array in `app.py` (line 24):

```python
NSE_STOCKS = [
    'RELIANCE', 'TCS', 'INFY',  # Add your stocks here
    # ... up to 500+ stocks
]
```

### Adjusting Update Frequency
In `app.py`, line 530 (WebSocket loop):

```python
await asyncio.sleep(8)  # Change to 5 for faster updates, 15 for slower
```

## 🐛 Troubleshooting

### Issue: "No data found" for stocks
**Solution:** Yahoo Finance may have rate limits. Solutions:
1. Reduce number of stocks in watchlist
2. Increase update interval to 10-15 seconds
3. Use polling instead of WebSocket

### Issue: WebSocket disconnects on Vercel
**Expected:** Vercel has 10s timeout for serverless functions
**Solution:** Use polling mode (see Option A above) or deploy to Railway/Render

### Issue: CORS errors
**Solution:** Backend automatically allows all origins. If issues persist:
```python
# In app.py, modify line 32:
allow_origins=["https://your-frontend-domain.com"]
```

### Issue: Slow performance
**Solutions:**
1. Reduce stocks in watchlist (max 10-15 for smooth updates)
2. Use Quick Scan (50 stocks) instead of Full Scan
3. Clear browser cache

## 📊 API Endpoints

### REST Endpoints

```
GET  /                    # Serve frontend
GET  /api/stocks         # Get list of available stocks
GET  /api/price/{symbol} # Get single stock price
POST /api/scan           # Scan multiple stocks
POST /api/analyze        # Deep analysis with beta
GET  /health            # Health check
```

### WebSocket Endpoint

```
WS /ws/prices           # Real-time price updates
```

**Message Format:**
```json
// Client sends:
{"symbols": ["RELIANCE", "TCS", "INFY"]}

// Server sends:
{
  "type": "price_update",
  "data": [
    {
      "symbol": "RELIANCE",
      "price": 2456.75,
      "change": 1.23,
      "volume": 5000000,
      "timestamp": "15:30:45",
      "status": "live"
    }
  ]
}
```

## 🎯 Usage Examples

### Example 1: Custom Watchlist
```javascript
// In browser console:
document.getElementById('watchlist').value = 'TATAMOTORS,MARUTI,M&M';
updateWatchlist();
```

### Example 2: Scan Custom Stocks
1. Switch to "Scanner" tab
2. Select "Custom List"
3. Enter symbols (one per line)
4. Click "Start Scan"

### Example 3: Quick Beta Check
1. Switch to "Beta Analyzer" tab
2. Enter symbol (e.g., "WIPRO")
3. Select period (1Y/3Y/5Y)
4. Click "Analyze"

## 🔐 Security Notes

- No API keys required (uses yfinance public data)
- No user authentication (stateless)
- CORS enabled for all origins (restrict in production)
- Rate limiting handled by caching (5s cache per stock)

## 📈 Performance Optimization

### Backend Caching
```python
# Already implemented in app.py
price_cache: Dict[str, Dict] = {}
last_fetch_time: Dict[str, datetime] = {}
```

### Frontend Optimization
- Vanilla JS (no framework overhead)
- CSS animations (GPU accelerated)
- Lazy loading of results
- Debounced WebSocket messages

## 🚨 Known Limitations

1. **Yahoo Finance Rate Limits**: May throttle after 50-100 requests/minute
   - Solution: Caching implemented, but avoid scanning 200+ stocks repeatedly

2. **Vercel WebSocket Timeout**: 10-second limit
   - Solution: Use polling mode or deploy to Railway/Render

3. **Market Hours**: Real-time data only during market hours (9:15 AM - 3:30 PM IST)
   - Outside hours: Shows previous close

4. **Data Accuracy**: Yahoo Finance data may have 5-15 minute delay
   - For true real-time, integrate with NSE API (requires authentication)

## 🔄 Alternative Deployment Options

### Railway.app (Better for WebSockets)
```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Deploy
railway up

# Your app will be at: https://your-project.up.railway.app
```

### Render.com
1. Go to render.com
2. New → Web Service
3. Connect GitHub repo
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

### Docker (Self-Hosted)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t stock-analyzer .
docker run -p 8000:8000 stock-analyzer
```

## 📝 License

MIT License - Free for personal and commercial use

## 🤝 Contributing

Feel free to fork and submit PRs!

## 📧 Support

For issues, create a GitHub issue or contact support.

---

**Built with ❤️ for the Indian stock market community**

**⚠️ Disclaimer:** This tool is for educational purposes only. Not financial advice. Always do your own research before investing.