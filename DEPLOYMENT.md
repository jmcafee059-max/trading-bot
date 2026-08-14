# Trading Bot Cloud Deployment Guide

This guide will help you deploy your trading bot to the cloud so you can access it from anywhere without your laptop.

## Recommended Cloud Platforms

### Option 1: Render (Free tier available)
- **Cost**: Free for web services
- **Pros**: Easy setup, automatic SSL, good for Flask apps
- **Cons**: Free tier spins down after inactivity

### Option 2: Railway (Free tier available)
- **Cost**: Free tier with $5/month credit
- **Pros**: Simple deployment, good for Python apps
- **Cons**: Limited free resources

### Option 3: Heroku (Paid)
- **Cost**: ~$5-7/month for Basic dyno
- **Pros**: Reliable, good documentation
- **Cons**: No free tier for web apps anymore

### Option 4: VPS (DigitalOcean, Linode)
- **Cost**: ~$4-6/month
- **Pros**: Full control, always running
- **Cons**: Requires server management

## Quick Deployment to Render (Recommended)

### Step 1: Push to GitHub
1. Create a GitHub repository
2. Push your trading bot files:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/trading-bot.git
   git push -u origin main
   ```

### Step 2: Deploy to Render
1. Go to [render.com](https://render.com)
2. Sign up/login
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Configure:
   - **Name**: trading-bot
   - **Branch**: main
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn api_server:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: Free (or Basic for $7/month)

### Step 3: Environment Variables
Add these environment variables in Render dashboard:
```
EXCHANGE_ID=binance
SYMBOL=SAND/USDT
TIMEFRAME=15m
STARTING_CAPITAL=18.0
CAPITAL_PERCENTAGE=90
RISK_PERCENTAGE=0.0
RSI_PERIOD=1
RSI_OVERBOUGHT=75
RSI_OVERSOLD=25
TAKE_PROFIT_PERCENT=1
```

### Step 4: Access Your Bot
- Render will provide a URL like: `https://trading-bot.onrender.com`
- Access this URL from any device
- Your bot will run 24/7

## Quick Deployment to Railway

### Step 1: Install Railway CLI
```bash
npm install -g @railway/cli
```

### Step 2: Login and Deploy
```bash
railway login
railway init
railway up
```

### Step 3: Add Environment Variables
```bash
railway variables set EXCHANGE_ID=binance
railway variables set SYMBOL=SAND/USDT
# Add all other variables...
```

## Important Notes

### Free Tier Limitations
- **Render**: Spins down after 15 minutes of inactivity (takes ~30s to wake up)
- **Railway**: Limited hours on free tier
- **Heroku**: No free tier for web apps

### Paid Tier Benefits
- Always running (24/7 trading)
- Faster response times
- Better reliability
- More resources

### Security Considerations
- Never commit real API keys to GitHub
- Use environment variables for sensitive data
- Consider adding password protection to the dashboard

### Monitoring
- Check logs regularly
- Monitor capital and trade count
- Set up alerts for significant losses

## Alternative: VPS Deployment

For full control and 24/7 operation:

### DigitalOcean Setup
1. Create a droplet ($4/month)
2. SSH into the server
3. Install dependencies:
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip nginx
   pip3 install -r requirements.txt
   ```
4. Set up systemd service
5. Configure nginx as reverse proxy
6. Add SSL with Let's Encrypt

## Current Files Ready for Deployment

✅ `api_server.py` - Flask application
✅ `simple_strategy_v2.py` - Trading strategy
✅ `requirements.txt` - Python dependencies
✅ `Procfile` - Process configuration
✅ `.env` - Environment variables (don't commit this!)

## Next Steps

1. Choose a platform (Render recommended for ease of use)
2. Push code to GitHub
3. Deploy following platform-specific instructions
4. Add environment variables
5. Test the deployed bot
6. Monitor from your phone

## Troubleshooting

**Bot not starting**: Check logs in platform dashboard
**API errors**: Verify exchange API keys (if using real trading)
**Connection issues**: Check firewall settings
**Memory issues**: Upgrade to paid tier

## Cost Summary

- **Render Free**: $0 (with spin-down)
- **Render Basic**: $7/month (always on)
- **Railway Free**: $0 (limited hours)
- **Railway Pro**: $5/month
- **Heroku Basic**: $5-7/month
- **DigitalOcean**: $4-6/month

For 24/7 trading, recommend Render Basic or DigitalOcean VPS.
