"""
pre_cache.py - Pre-cache popular airports periodically to populate disk storage.
"""

import sys
import time
from pathlib import Path

# Add project root to python path to support execution from any location
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.fetcher import fetch_metar, fetch_taf
from src.agent import analyze_metar_orchestrator
from src.cache_manager import weather_cache

POPULAR_AIRPORTS = [
    # India
    "VABB", "VIDP", "VOBL", "VOMM", "VECC", "VAAH", "VAPO", "VIAR",
    # US
    "KJFK", "KLAX", "KORD", "KATL", "KDFW", "KDEN", "KSFO", "KMIA",
    # Europe
    "EGLL", "EGKK", "LFPG", "EDDF", "EHAM", "LEMD", "LIRF",
    # Middle East
    "OMDB", "OTHH", "OEJN", "OKBK",
    # Asia
    "VHHH", "WSSS", "WMKK", "VTBS", "RJTT", "RKSI", "ZBAA",
    # Oceania
    "YSSY", "YBBN", "NZAA",
]

def pre_cache_airports():
    """Pre-fetch and cache popular airports"""
    print("=" * 60)
    print("🔄 STARTING METAR/TAF PRE-CACHING SUITE...")
    print("=" * 60)
    
    start_time = time.time()
    success_count = 0
    fail_count = 0
    
    for i, icao in enumerate(POPULAR_AIRPORTS, start=1):
        print(f"[{i}/{len(POPULAR_AIRPORTS)}] Processing {icao}...")
        try:
            # 1. Fetch live raw weather
            raw_metar = fetch_metar(icao)
            if not raw_metar or "Error" in raw_metar or "Exception" in raw_metar:
                print(f"  ⚠️ Skipped: Live METAR data unavailable for {icao}")
                fail_count += 1
                continue
                
            raw_taf = fetch_taf(icao)
            if not raw_taf or "Error" in raw_taf or "Exception" in raw_taf:
                raw_taf = ""
            
            # 2. Run through orchestrator (force new analysis, but save to cache)
            # This automatically populates the weather_cache directory
            analysis = analyze_metar_orchestrator(raw_metar, raw_taf, use_cache=True)
            
            if analysis.get("status") in ["success", "partial"]:
                print(f"  ✅ Cached successfully: {icao} (Provider: {analysis.get('provider')})")
                success_count += 1
            else:
                print(f"  ❌ Analysis returned status '{analysis.get('status')}' for {icao}")
                fail_count += 1
                
            # 3. Rate Limit Cooldown spacing
            time.sleep(2.0)
            
        except Exception as e:
            print(f"  ⚠️ Failed to pre-cache {icao}: {e}")
            fail_count += 1
            time.sleep(2.0)
            
    duration = time.time() - start_time
    print("=" * 60)
    print("🎉 PRE-CACHING COMPLETED")
    print(f"   ⏱️  Total Duration: {duration:.1f} seconds")
    print(f"   ✅ Successful Runs: {success_count}")
    print(f"   ❌ Failed/Skipped: {fail_count}")
    
    stats = weather_cache.get_stats()
    print(f"   📦 Cache Status: {stats.get('memory_entries')} memory / {stats.get('disk_entries')} disk entries total.")
    print("=" * 60)

# Run if executed directly
if __name__ == "__main__":
    pre_cache_airports()
