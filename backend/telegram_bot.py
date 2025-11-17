"""
Telegram Bot for Real-time Toxicity Detection
Integrates with Detox ML model for live message analysis
"""

import asyncio
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import requests
import config
from backend.telegram_store import telegram_store

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# API endpoint for predictions
PREDICT_API_URL = "http://localhost:5000/api/v1/predict"


class ToxicityBot:
    """Telegram bot for toxicity detection"""
    
    def __init__(self, token):
        self.token = token
        self.app = None
        self.is_running = False
        
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        welcome_message = """
🤖 **Welcome to Detox Toxicity Detector Bot!**

I can analyze text messages for toxic content in real-time.

**Commands:**
/start - Show this welcome message
/analyze <text> - Analyze specific text
/stats - Show analysis statistics
/help - Show help information
/moderate on/off - Toggle auto-moderation

**How to use:**
Just send me any message and I'll analyze it instantly!

**Toxicity Levels:**
• MINIMAL (0-20%) - Clean content
• LOW (20-40%) - Slightly negative
• MEDIUM (40-60%) - Moderately toxic
• HIGH (60-80%) - Very toxic
• VERY HIGH (80-100%) - Extremely toxic

Powered by PySpark MLlib + Lexicon Features (90.19% accuracy)
"""
        await update.message.reply_text(welcome_message, parse_mode='Markdown')
        logger.info(f"User {update.effective_user.id} started the bot")
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = """
📚 **Help - Toxicity Detector Bot**

**Available Commands:**
• /start - Welcome message
• /analyze <text> - Analyze specific text
• /stats - View analysis statistics
• /help - Show this help message
• /moderate on/off - Toggle auto-moderation

**How it works:**
1. Send any text message to the bot
2. The ML model analyzes toxicity (TF-IDF + Lexicon)
3. You receive instant feedback with:
   - Toxicity prediction (Toxic/Non-toxic)
   - Toxicity score (0.0 to 1.0)
   - Severity level (MINIMAL to VERY HIGH)
   - Processing time

**Features:**
✅ 90.19% accuracy
✅ <1 second response time
✅ Lexicon-based detection (80+ toxic words)
✅ Real-time analysis
✅ Group chat support

**Support:** @YourUsername
"""
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command"""
        stats = telegram_store.get_stats()
        
        stats_text = f"""
📊 **Bot Statistics**

**Total Messages Analyzed:** {stats['total_analyzed']}
**Toxic Messages:** {stats['total_toxic']} ({stats.get('toxic_percentage', 0):.1f}%)
**Non-toxic Messages:** {stats['total_non_toxic']} ({stats.get('non_toxic_percentage', 0):.1f}%)

**By Severity Level:**
• MINIMAL: {stats['by_level'].get('MINIMAL', 0)}
• LOW: {stats['by_level'].get('LOW', 0)}
• MEDIUM: {stats['by_level'].get('MEDIUM', 0)}
• HIGH: {stats['by_level'].get('HIGH', 0)}
• VERY HIGH: {stats['by_level'].get('VERY_HIGH', 0)}

**Top Users (by messages):**
"""
        # Add top 5 users
        users_sorted = sorted(
            stats['by_user'].items(), 
            key=lambda x: x[1]['total'], 
            reverse=True
        )[:5]
        
        for i, (user, user_stats) in enumerate(users_sorted, 1):
            toxic_pct = (user_stats['toxic'] / user_stats['total'] * 100) if user_stats['total'] > 0 else 0
            stats_text += f"\n{i}. {user}: {user_stats['total']} msgs ({toxic_pct:.1f}% toxic)"
        
        if not users_sorted:
            stats_text += "\nNo users yet!"
        
        await update.message.reply_text(stats_text, parse_mode='Markdown')
        logger.info(f"User {update.effective_user.id} requested stats")
    
    async def analyze_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /analyze command"""
        if not context.args:
            await update.message.reply_text(
                "❌ Please provide text to analyze.\n\nExample: /analyze This is a test message"
            )
            return
        
        text = ' '.join(context.args)
        await self.analyze_message(update, text)
    
    async def moderate_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /moderate command"""
        if not context.args or context.args[0].lower() not in ['on', 'off']:
            await update.message.reply_text(
                "❌ Usage: /moderate on|off\n\nExample: /moderate on"
            )
            return
        
        enable = context.args[0].lower() == 'on'
        config.TELEGRAM_AUTO_MODERATE = enable
        
        status = "enabled ✅" if enable else "disabled ❌"
        await update.message.reply_text(
            f"🛡️ Auto-moderation {status}\n\n"
            f"Threshold: {config.TELEGRAM_MODERATE_THRESHOLD * 100}% toxicity\n"
            f"Toxic messages will {'be deleted' if enable else 'NOT be deleted'}."
        )
        logger.info(f"User {update.effective_user.id} {'enabled' if enable else 'disabled'} auto-moderation")
    
    async def analyze_message(self, update: Update, text: str):
        """Analyze a message for toxicity"""
        try:
            # Send "analyzing..." message
            analyzing_msg = await update.message.reply_text("🔍 Analyzing message...")
            
            # Call prediction API
            response = requests.post(
                PREDICT_API_URL,
                json={"text": text},
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code != 200:
                await analyzing_msg.edit_text(
                    "❌ Error: Model service unavailable. Make sure the backend is running."
                )
                return
            
            result = response.json()
            
            if not result.get('success'):
                await analyzing_msg.edit_text(
                    f"❌ Error: {result.get('error', 'Unknown error')}"
                )
                return
            
            # Store in telegram_store
            user = update.effective_user
            telegram_store.add_message(
                user_id=user.id,
                username=user.username,
                first_name=user.first_name,
                chat_id=update.effective_chat.id,
                message_text=text,
                analysis_result=result
            )
            telegram_store.update_last_message_time()
            
            # Format result
            prediction = result.get('prediction', 'Unknown')
            score = result.get('toxicity_score', 0.0)
            level = result.get('level', 'UNKNOWN')
            processing_time = result.get('processing_time_ms', 0)
            lexicon_severity = result.get('lexicon_severity', 'Unknown')
            
            # Choose emoji based on level
            emoji_map = {
                'MINIMAL': '✅',
                'LOW': '🟡',
                'MEDIUM': '🟠',
                'HIGH': '🔴',
                'VERY_HIGH': '⚠️'
            }
            emoji = emoji_map.get(level, '❓')
            
            result_text = f"""
{emoji} **Analysis Complete**

**Prediction:** {prediction}
**Toxicity Score:** {score:.4f} ({score*100:.2f}%)
**Severity Level:** {level}
**Lexicon Severity:** {lexicon_severity}
**Processing Time:** {processing_time:.2f}ms

**Message:** "{text[:100]}{'...' if len(text) > 100 else ''}"
"""
            
            await analyzing_msg.edit_text(result_text, parse_mode='Markdown')
            
            # Auto-moderation
            if config.TELEGRAM_AUTO_MODERATE and score >= config.TELEGRAM_MODERATE_THRESHOLD:
                try:
                    await update.message.delete()
                    warning = await update.message.reply_text(
                        f"⚠️ Message deleted due to high toxicity ({score*100:.1f}%)\n"
                        f"Please keep the conversation respectful! 🙏"
                    )
                    # Delete warning after 10 seconds
                    await asyncio.sleep(10)
                    await warning.delete()
                    logger.warning(f"Deleted toxic message from user {user.id}: {text[:50]}")
                except Exception as e:
                    logger.error(f"Could not delete message: {e}")
            
            # Warning for high toxicity (even if auto-moderate is off)
            elif score >= 0.7 and config.TELEGRAM_WARN_USERS:
                await update.message.reply_text(
                    "⚠️ **Warning:** Your message contains toxic content. "
                    "Please be respectful in your communication. 🙏"
                )
            
        except requests.Timeout:
            await update.message.reply_text(
                "❌ Request timeout. The model service took too long to respond."
            )
        except requests.ConnectionError:
            await update.message.reply_text(
                "❌ Connection error. Make sure the backend server is running on http://localhost:5000"
            )
        except Exception as e:
            logger.error(f"Error analyzing message: {e}")
            await update.message.reply_text(
                f"❌ An error occurred: {str(e)}"
            )
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle regular messages"""
        text = update.message.text
        await self.analyze_message(update, text)
    
    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors"""
        logger.error(f"Update {update} caused error {context.error}")
        telegram_store.set_bot_status(True, str(context.error))
    
    def run(self):
        """Run the bot"""
        try:
            logger.info(f"Starting Telegram bot...")
            telegram_store.set_bot_status(True)
            
            # Create application
            self.app = Application.builder().token(self.token).build()
            
            # Add handlers
            self.app.add_handler(CommandHandler("start", self.start_command))
            self.app.add_handler(CommandHandler("help", self.help_command))
            self.app.add_handler(CommandHandler("stats", self.stats_command))
            self.app.add_handler(CommandHandler("analyze", self.analyze_command))
            self.app.add_handler(CommandHandler("moderate", self.moderate_command))
            self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
            self.app.add_error_handler(self.error_handler)
            
            logger.info("✓ Bot handlers registered")
            logger.info("✓ Bot is running! Press Ctrl+C to stop.")
            
            # Run bot
            self.is_running = True
            self.app.run_polling(allowed_updates=Update.ALL_TYPES)
            
        except Exception as e:
            logger.error(f"Error running bot: {e}")
            telegram_store.set_bot_status(False, str(e))
            raise
        finally:
            self.is_running = False
            telegram_store.set_bot_status(False)
            logger.info("Bot stopped")
    
    def stop(self):
        """Stop the bot"""
        if self.app and self.is_running:
            self.app.stop()
            self.is_running = False
            telegram_store.set_bot_status(False)
            logger.info("Bot stop requested")


# Main entry point
if __name__ == "__main__":
    token = config.TELEGRAM_BOT_TOKEN
    
    if token == 'YOUR_BOT_TOKEN_HERE' or not token:
        print("\n" + "="*60)
        print("❌ ERROR: Telegram Bot Token not configured!")
        print("="*60)
        print("\n📱 Please follow these steps to set up your bot:\n")
        print("1. Open Telegram and search for @BotFather")
        print("2. Send: /newbot")
        print("3. Follow instructions to create your bot")
        print("4. Copy the API token you receive")
        print("5. Set environment variable: export TELEGRAM_BOT_TOKEN='your-token-here'")
        print("   OR edit config.py and replace YOUR_BOT_TOKEN_HERE")
        print("\n" + "="*60 + "\n")
        exit(1)
    
    bot = ToxicityBot(token)
    
    try:
        bot.run()
    except KeyboardInterrupt:
        print("\n\n👋 Stopping bot...")
        bot.stop()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        exit(1)
