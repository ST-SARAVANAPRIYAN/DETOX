"""
Production API Endpoints - Quick Win #2
=======================================
Fast, secure endpoints for real-time toxicity detection
Includes rate limiting, authentication, and monitoring
"""

from flask import Blueprint, request, jsonify
from functools import wraps
import time
from collections import defaultdict
from threading import Lock
import logging

logger = logging.getLogger(__name__)

# Create API blueprint
api_v1 = Blueprint('api_v1', __name__, url_prefix='/api/v1')

# Rate limiting
class RateLimiter:
    def __init__(self):
        self.requests = defaultdict(list)
        self.lock = Lock()
    
    def is_allowed(self, client_ip: str, limit: int = 100, window: int = 60) -> bool:
        """
        Check if request is allowed based on rate limit
        
        Args:
            client_ip: Client IP address
            limit: Max requests per window
            window: Time window in seconds
        """
        with self.lock:
            now = time.time()
            
            # Remove old requests
            self.requests[client_ip] = [
                req_time for req_time in self.requests[client_ip]
                if now - req_time < window
            ]
            
            # Check limit
            if len(self.requests[client_ip]) >= limit:
                return False
            
            # Add new request
            self.requests[client_ip].append(now)
            return True
    
    def get_remaining(self, client_ip: str, limit: int = 100, window: int = 60) -> int:
        """Get remaining requests in current window"""
        with self.lock:
            now = time.time()
            self.requests[client_ip] = [
                req_time for req_time in self.requests[client_ip]
                if now - req_time < window
            ]
            return max(0, limit - len(self.requests[client_ip]))


rate_limiter = RateLimiter()

def rate_limit(limit=100, window=60):
    """Rate limiting decorator"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            client_ip = request.remote_addr
            
            if not rate_limiter.is_allowed(client_ip, limit, window):
                remaining = rate_limiter.get_remaining(client_ip, limit, window)
                return jsonify({
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {limit} requests per {window} seconds",
                    "remaining": remaining,
                    "retry_after": window
                }), 429
            
            # Add rate limit headers
            response = f(*args, **kwargs)
            if isinstance(response, tuple):
                response_data, status_code = response
            else:
                response_data = response
                status_code = 200
            
            remaining = rate_limiter.get_remaining(client_ip, limit, window)
            
            if isinstance(response_data, dict):
                response_data = jsonify(response_data)
            
            response_data.headers['X-RateLimit-Limit'] = str(limit)
            response_data.headers['X-RateLimit-Remaining'] = str(remaining)
            response_data.headers['X-RateLimit-Window'] = str(window)
            
            return response_data, status_code
        
        return wrapper
    return decorator


def require_api_key(f):
    """API key authentication decorator"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        
        # For development: allow requests without key
        # In production: validate against database
        if not api_key:
            return jsonify({
                "error": "Missing API key",
                "message": "Include X-API-Key header with your API key"
            }), 401
        
        # TODO: Validate API key against database
        # For now, accept any non-empty key
        if len(api_key) < 10:
            return jsonify({
                "error": "Invalid API key",
                "message": "API key must be at least 10 characters"
            }), 401
        
        return f(*args, **kwargs)
    
    return wrapper


@api_v1.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    GET /api/v1/health
    """
    from backend.model_cache import model_cache
    
    try:
        stats = model_cache.get_stats()
        return jsonify({
            "status": "healthy",
            "model_loaded": stats["model_loaded"],
            "spark_active": stats["spark_active"],
            "spark_version": stats["spark_version"],
            "timestamp": time.time()
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }), 503


@api_v1.route('/predict', methods=['POST'])
@rate_limit(limit=100, window=60)  # 100 requests per minute
def predict_single():
    """
    Predict toxicity for a single message
    
    POST /api/v1/predict
    Content-Type: application/json
    X-API-Key: your-api-key (optional for now)
    
    Request Body:
    {
        "text": "Your message here"
    }
    
    Response:
    {
        "prediction": "Toxic" or "Non-toxic",
        "toxicity_score": 0.85,
        "level": "HIGH",
        "latency_ms": 45.23,
        "success": true
    }
    """
    from backend.model_cache import model_cache
    
    try:
        # Validate request
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                "error": "Invalid request",
                "message": "Request body must include 'text' field"
            }), 400
        
        text = data['text']
        
        # Validate text
        if not isinstance(text, str) or len(text) == 0:
            return jsonify({
                "error": "Invalid text",
                "message": "Text must be a non-empty string"
            }), 400
        
        if len(text) > 5000:
            return jsonify({
                "error": "Text too long",
                "message": "Text must be less than 5000 characters"
            }), 400
        
        # Predict
        result = model_cache.predict_single(text)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e),
            "success": False
        }), 500


@api_v1.route('/predict/batch', methods=['POST'])
@rate_limit(limit=20, window=60)  # 20 batch requests per minute
def predict_batch():
    """
    Predict toxicity for multiple messages at once (faster than individual requests)
    
    POST /api/v1/predict/batch
    Content-Type: application/json
    X-API-Key: your-api-key (optional for now)
    
    Request Body:
    {
        "texts": ["Message 1", "Message 2", "Message 3"]
    }
    
    Response:
    {
        "predictions": [
            {
                "text": "Message 1...",
                "prediction": "Non-toxic",
                "toxicity_score": 0.12,
                "level": "MINIMAL"
            },
            ...
        ],
        "total": 3,
        "total_latency_ms": 120.45,
        "avg_latency_ms": 40.15,
        "success": true
    }
    """
    from backend.model_cache import model_cache
    
    try:
        # Validate request
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({
                "error": "Invalid request",
                "message": "Request body must include 'texts' field (array of strings)"
            }), 400
        
        texts = data['texts']
        
        # Validate texts
        if not isinstance(texts, list):
            return jsonify({
                "error": "Invalid texts",
                "message": "texts must be an array"
            }), 400
        
        if len(texts) == 0:
            return jsonify({
                "error": "Empty texts",
                "message": "texts array cannot be empty"
            }), 400
        
        if len(texts) > 100:
            return jsonify({
                "error": "Too many texts",
                "message": "Maximum 100 texts per batch request"
            }), 400
        
        # Validate each text
        for i, text in enumerate(texts):
            if not isinstance(text, str) or len(text) == 0:
                return jsonify({
                    "error": f"Invalid text at index {i}",
                    "message": "Each text must be a non-empty string"
                }), 400
            
            if len(text) > 5000:
                return jsonify({
                    "error": f"Text too long at index {i}",
                    "message": "Each text must be less than 5000 characters"
                }), 400
        
        # Predict
        result = model_cache.predict_batch(texts)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e),
            "success": False
        }), 500


@api_v1.route('/stats', methods=['GET'])
def get_stats():
    """
    Get API statistics
    GET /api/v1/stats
    """
    from backend.model_cache import model_cache
    
    try:
        stats = model_cache.get_stats()
        return jsonify({
            "api_version": "1.0.0",
            "model_stats": stats,
            "rate_limits": {
                "predict_single": "100 requests/minute",
                "predict_batch": "20 requests/minute (max 100 texts per request)"
            },
            "max_text_length": 5000
        }), 200
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500
