"""
cache_manager.py - Prevents redundant API calls with thread-safe memory & disk caching.
"""

import hashlib
import json
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any

class WeatherCache:
    """
    Cache weather analyses to prevent duplicate API calls.
    Multiple users searching the same airport with the same METAR = 1 API call.
    """
    
    def __init__(self, cache_dir: str = "weather_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
    
    def _get_cache_key(self, icao: str, raw_metar: str) -> str:
        """Generate unique cache key from airport + weather data"""
        normalized_icao = icao.strip().upper()
        normalized_metar = raw_metar.strip().upper()
        content = f"{normalized_icao}:{normalized_metar}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, icao: str, raw_metar: str, max_age_seconds: int = 300) -> Optional[Dict]:
        """
        Get cached analysis if available and fresh.
        Default: 5 minutes cache (300 seconds)
        """
        cache_key = self._get_cache_key(icao, raw_metar)
        
        with self.lock:
            # 1. Check memory cache first (fastest)
            if cache_key in self.memory_cache:
                cached = self.memory_cache[cache_key]
                age = time.time() - cached['timestamp']
                if age < max_age_seconds:
                    print(f"⚡ Memory cache hit for {icao} ({age:.0f}s old)")
                    return cached['data']
                else:
                    # Stale entry, delete it from memory cache
                    del self.memory_cache[cache_key]
            
            # 2. Check disk cache (survives app restarts)
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cached = json.load(f)
                    age = time.time() - cached['timestamp']
                    if age < max_age_seconds:
                        print(f"💾 Disk cache hit for {icao} ({age:.0f}s old)")
                        # Promote to memory cache
                        self.memory_cache[cache_key] = cached
                        return cached['data']
                    else:
                        # Stale entry, clean up disk
                        try:
                            cache_file.unlink()
                        except OSError:
                            pass
                except Exception as e:
                    print(f"⚠️ Error reading disk cache for {icao}: {e}")
        
        return None
    
    def set(self, icao: str, raw_metar: str, data: Dict):
        """Cache analysis result"""
        cache_key = self._get_cache_key(icao, raw_metar)
        
        cached = {
            'timestamp': time.time(),
            'data': data,
            'icao': icao.strip().upper()
        }
        
        with self.lock:
            # Store in memory
            self.memory_cache[cache_key] = cached
            
            # Store on disk
            cache_file = self.cache_dir / f"{cache_key}.json"
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(cached, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"⚠️ Error writing disk cache for {icao}: {e}")
            
            # Cleanup old memory cache entries if size exceeds 50
            if len(self.memory_cache) > 50:
                # Get the oldest 10 entries and delete them
                oldest = sorted(self.memory_cache.items(), 
                              key=lambda x: x[1]['timestamp'])[:10]
                for key, _ in oldest:
                    del self.memory_cache[key]
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        with self.lock:
            try:
                disk_count = len(list(self.cache_dir.glob("*.json")))
            except Exception:
                disk_count = 0
            
            return {
                "memory_entries": len(self.memory_cache),
                "disk_entries": disk_count,
            }

# Global cache instance
weather_cache = WeatherCache()
