# 📱 Telegram Bot Setup Guide

## Overview
The Detox Telegram Bot provides real-time toxicity detection for Telegram messages. Users can send messages to the bot for instant analysis, or add the bot to groups for automated moderation.

## Features
- ✅ Real-time toxicity analysis (<1s response)
- ✅ 90.19% accuracy with lexicon features
- ✅ Auto-moderation (optional)
- ✅ User warnings for toxic content
- ✅ Group chat support
- ✅ Statistics tracking
- ✅ Web dashboard integration

---

## 🚀 Quick Setup (5 minutes)

### Step 1: Create Your Telegram Bot

1. **Open Telegram** on your phone or desktop

2. **Search for** `@BotFather` (official Telegram bot creator)

3. **Send command:** `/newbot`

4. **Follow the prompts:**
   - Enter a name for your bot (e.g., "Detox Toxicity Detector")
   - Enter a username ending in "bot" (e.g., "detox_toxicity_bot")

5. **Copy the API Token** you receive (looks like: `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)

### Step 2: Configure the Bot

**Option A: Environment Variable (Recommended)**
```bash
export TELEGRAM_BOT_TOKEN='your-token-here'
```

**Option B: Edit config.py**
```python
# In config.py, replace:
TELEGRAM_BOT_TOKEN = 'YOUR_BOT_TOKEN_HERE'

# With:
TELEGRAM_BOT_TOKEN = '123456789:ABCdefGHIjklMNOpqrsTUVwxyz'
```

### Step 3: Install Dependencies

```bash
# Activate virtual environment
source venv/bin/activate

# Install Telegram bot library (already in requirements.txt)
pip install python-telegram-bot==20.7
```

### Step 4: Run the Bot

**Terminal 1: Start Backend API (Required)**
```bash
cd /path/to/ssfproject
source venv/bin/activate
python backend/app.py
```
Wait for: `✓ Production API v1 registered at /api/v1`

**Terminal 2: Start Telegram Bot**
```bash
cd /path/to/ssfproject
source venv/bin/activate
python backend/telegram_bot.py
```
Wait for: `✓ Bot is running! Press Ctrl+C to stop.`

**Terminal 3: Start Frontend (Optional - for web dashboard)**
```bash
cd /path/to/ssfproject/frontend
npm run dev
```

---

## 🎮 Using the Bot

### Personal Chat

1. **Find your bot** on Telegram (search for username you created)

2. **Start the bot:**
   ```
   /start
   ```

3. **Analyze text:**
   - Simply send any message
   - Or use: `/analyze Your message here`

4. **View statistics:**
   ```
   /stats
   ```

5. **Get help:**
   ```
   /help
   ```

### Group Chat

1. **Add bot to group:**
   - Open group chat
   - Click ⋮ menu → Add members
   - Search for your bot
   - Add it

2. **Give admin permissions (for auto-moderation):**
   - Group info → Administrators
   - Add bot as admin
   - Enable "Delete messages" permission

3. **Enable auto-moderation:**
   ```
   /moderate on
   ```

4. **Messages will be analyzed automatically**

---

## ⚙️ Configuration

Edit `config.py` to customize bot behavior:

```python
# Auto-delete toxic messages?
TELEGRAM_AUTO_MODERATE = False  # Set True to enable

# Toxicity threshold for deletion
TELEGRAM_MODERATE_THRESHOLD = 0.8  # 80% toxicity

# Maximum messages to keep in memory
TELEGRAM_MAX_MESSAGE_HISTORY = 100

# Send warnings to users?
TELEGRAM_WARN_USERS = True
```

---

## 🌐 Web Dashboard

Access the live feed dashboard at: http://localhost:5174

**Features:**
- 📊 Real-time message feed from Telegram
- 📈 Statistics dashboard
- 🎨 Color-coded toxicity levels
- 📥 Export analysis results
- 🔄 Auto-refresh every 5 seconds

---

## 🎯 Commands Reference

| Command | Description | Example |
|---------|-------------|---------|
| `/start` | Welcome message | `/start` |
| `/help` | Show help | `/help` |
| `/analyze <text>` | Analyze specific text | `/analyze test message` |
| `/stats` | View statistics | `/stats` |
| `/moderate on/off` | Toggle auto-moderation | `/moderate on` |

---

## 📊 Response Format

When you send a message, you'll receive:

```
✅ Analysis Complete

Prediction: Non-toxic
Toxicity Score: 0.1234 (12.34%)
Severity Level: MINIMAL
Lexicon Severity: Clean
Processing Time: 45.67ms

Message: "Your text here..."
```

**Severity Levels:**
- ✅ **MINIMAL** (0-20%) - Clean content
- 🟡 **LOW** (20-40%) - Slightly negative
- 🟠 **MEDIUM** (40-60%) - Moderately toxic
- 🔴 **HIGH** (60-80%) - Very toxic
- ⚠️ **VERY HIGH** (80-100%) - Extremely toxic

---

## 🛡️ Auto-Moderation

When enabled (`/moderate on`), the bot will:
1. ✅ Analyze every message
2. ⚠️ Delete messages above threshold (default: 80%)
3. 📢 Send warning to user
4. 📊 Track violations per user
5. 🗑️ Auto-delete warning after 10 seconds

**Requirements:**
- Bot must be admin in group
- Bot needs "Delete messages" permission

---

## 🔧 Troubleshooting

### Bot doesn't respond
- ✅ Check backend API is running (`python backend/app.py`)
- ✅ Check bot is running (`python backend/telegram_bot.py`)
- ✅ Verify token is correct in config.py
- ✅ Check console for error messages

### "Model service unavailable" error
- ✅ Start backend API first
- ✅ Wait for model to load (check console: "✓ Production API v1 registered")
- ✅ Test API: `curl http://localhost:5000/api/v1/health`

### Can't delete messages in group
- ✅ Make bot admin
- ✅ Enable "Delete messages" permission
- ✅ Check `/moderate on` is enabled

### Bot stops after some time
- ✅ Check terminal for errors
- ✅ Restart: `python backend/telegram_bot.py`
- ✅ Check network connection
- ✅ Verify Telegram API access (no firewall blocking)

---

## 🚀 Production Deployment

### Using systemd (Linux)

1. **Create service file:** `/etc/systemd/system/detox-telegram.service`
```ini
[Unit]
Description=Detox Telegram Bot
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/ssfproject
Environment="TELEGRAM_BOT_TOKEN=your-token-here"
ExecStart=/path/to/ssfproject/venv/bin/python backend/telegram_bot.py
Restart=always

[Install]
WantedBy=multi-user.target
```

2. **Enable and start:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable detox-telegram
sudo systemctl start detox-telegram
sudo systemctl status detox-telegram
```

### Using Docker

```dockerfile
FROM python:3.13
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENV TELEGRAM_BOT_TOKEN=your-token-here
CMD ["python", "backend/telegram_bot.py"]
```

```bash
docker build -t detox-telegram .
docker run -d --name detox-bot -e TELEGRAM_BOT_TOKEN='your-token' detox-telegram
```

---

## 📝 API Integration

The bot data is accessible via REST API:

### Get Recent Messages
```bash
curl http://localhost:5000/api/telegram/messages?limit=50
```

### Get Statistics
```bash
curl http://localhost:5000/api/telegram/stats
```

### Clear Messages
```bash
curl -X POST http://localhost:5000/api/telegram/clear
```

### Reset Statistics
```bash
curl -X POST http://localhost:5000/api/telegram/reset-stats
```

---

## 🎓 For Academic Projects

**Presentation Demo Script:**

1. ✅ Show bot creation with @BotFather
2. ✅ Send clean message → Show MINIMAL result
3. ✅ Send toxic message → Show VERY HIGH result
4. ✅ Show `/stats` → Display analytics
5. ✅ Open web dashboard → Show real-time feed
6. ✅ Enable `/moderate on` → Demonstrate auto-deletion
7. ✅ Export analysis results

---

## 📞 Support

- **Issues:** Check console output for error messages
- **Logs:** Bot logs all activities to console
- **API Status:** http://localhost:5000/api/v1/health
- **Model Info:** http://localhost:5000/api/v1/info

---

## ✨ Advanced Features

### Custom Commands (Edit telegram_bot.py)

Add new commands:
```python
async def custom_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Custom response")

# In run() method:
self.app.add_handler(CommandHandler("custom", self.custom_command))
```

### Database Integration

Replace `telegram_store` with database:
```python
# Use SQLite/PostgreSQL/MongoDB instead of in-memory storage
# For persistent analytics and reporting
```

### Multi-language Support

Add translation dictionaries for bot messages.

---

## 🎉 You're Done!

Your Telegram bot is now running and detecting toxic content in real-time!

**Next Steps:**
- Add bot to your groups
- Monitor analytics on web dashboard
- Customize moderation thresholds
- Export reports for analysis

**Questions?** Check logs in terminal or open an issue.

---

**Project:** Detox - Toxicity Detection System
**Technology:** PySpark MLlib + Telegram Bot API
**Accuracy:** 90.19% with lexicon features
**Response Time:** <1 second
