"""
Telegram Bot Message Store
Stores analyzed messages in memory for real-time display
"""

from datetime import datetime
from collections import deque
from threading import Lock
import config

class TelegramMessageStore:
    """Thread-safe store for Telegram bot messages and analysis results"""
    
    def __init__(self, max_messages=None):
        self.max_messages = max_messages or config.TELEGRAM_MAX_MESSAGE_HISTORY
        self.messages = deque(maxlen=self.max_messages)
        self.lock = Lock()
        self.stats = {
            'total_analyzed': 0,
            'total_toxic': 0,
            'total_non_toxic': 0,
            'by_level': {
                'MINIMAL': 0,
                'LOW': 0,
                'MEDIUM': 0,
                'HIGH': 0,
                'VERY_HIGH': 0
            },
            'by_user': {}
        }
        self.bot_status = {
            'running': False,
            'started_at': None,
            'last_message_at': None,
            'error': None
        }
    
    def add_message(self, user_id, username, first_name, chat_id, message_text, analysis_result):
        """Add a new analyzed message to the store"""
        with self.lock:
            message_data = {
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'username': username,
                'first_name': first_name,
                'chat_id': chat_id,
                'text': message_text[:200],  # Truncate long messages
                'full_text': message_text,
                'prediction': analysis_result.get('prediction'),
                'toxicity_score': analysis_result.get('toxicity_score'),
                'level': analysis_result.get('level'),
                'lexicon_severity': analysis_result.get('lexicon_severity'),
                'processing_time_ms': analysis_result.get('processing_time_ms')
            }
            
            self.messages.appendleft(message_data)  # Add to front (most recent first)
            
            # Update stats
            self.stats['total_analyzed'] += 1
            self.stats['last_message_at'] = datetime.now().isoformat()
            
            prediction = analysis_result.get('prediction', '')
            if 'toxic' in prediction.lower() and prediction.lower() != 'non-toxic':
                self.stats['total_toxic'] += 1
            else:
                self.stats['total_non_toxic'] += 1
            
            level = analysis_result.get('level', 'MINIMAL')
            self.stats['by_level'][level] = self.stats['by_level'].get(level, 0) + 1
            
            # Track per-user stats
            user_key = f"{username or first_name} ({user_id})"
            if user_key not in self.stats['by_user']:
                self.stats['by_user'][user_key] = {
                    'total': 0,
                    'toxic': 0,
                    'max_toxicity': 0.0
                }
            
            self.stats['by_user'][user_key]['total'] += 1
            if 'toxic' in prediction.lower() and prediction.lower() != 'non-toxic':
                self.stats['by_user'][user_key]['toxic'] += 1
            
            score = analysis_result.get('toxicity_score', 0.0)
            if score > self.stats['by_user'][user_key]['max_toxicity']:
                self.stats['by_user'][user_key]['max_toxicity'] = score
            
            return message_data
    
    def get_recent_messages(self, limit=50):
        """Get recent analyzed messages"""
        with self.lock:
            return list(self.messages)[:limit]
    
    def get_stats(self):
        """Get aggregated statistics"""
        with self.lock:
            stats = self.stats.copy()
            
            # Calculate percentages
            if stats['total_analyzed'] > 0:
                stats['toxic_percentage'] = (stats['total_toxic'] / stats['total_analyzed']) * 100
                stats['non_toxic_percentage'] = (stats['total_non_toxic'] / stats['total_analyzed']) * 100
            else:
                stats['toxic_percentage'] = 0.0
                stats['non_toxic_percentage'] = 0.0
            
            # Add bot status
            stats['bot_status'] = self.bot_status.copy()
            
            return stats
    
    def clear_messages(self):
        """Clear all stored messages"""
        with self.lock:
            self.messages.clear()
    
    def reset_stats(self):
        """Reset all statistics"""
        with self.lock:
            self.stats = {
                'total_analyzed': 0,
                'total_toxic': 0,
                'total_non_toxic': 0,
                'by_level': {
                    'MINIMAL': 0,
                    'LOW': 0,
                    'MEDIUM': 0,
                    'HIGH': 0,
                    'VERY_HIGH': 0
                },
                'by_user': {}
            }
    
    def set_bot_status(self, running, error=None):
        """Update bot status"""
        with self.lock:
            self.bot_status['running'] = running
            self.bot_status['error'] = error
            if running and self.bot_status['started_at'] is None:
                self.bot_status['started_at'] = datetime.now().isoformat()
            elif not running:
                self.bot_status['started_at'] = None
    
    def update_last_message_time(self):
        """Update the timestamp of the last received message"""
        with self.lock:
            self.bot_status['last_message_at'] = datetime.now().isoformat()


# Global instance
telegram_store = TelegramMessageStore()
