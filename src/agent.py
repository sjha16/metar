"""
Aviation Weather Briefing Parser
Purely deterministic regex-based METAR/TAF parser.
No AI dependencies or API keys required.
"""

import os
import re
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from functools import lru_cache

# Import caching layers
from src.cache_manager import weather_cache

# ============================================
# RUNWAY CALCULATION HELPERS
# ============================================

def _calculate_active_runway(wind_direction_deg: Optional[int], runways: list) -> Tuple[Optional[str], str]:
    """
    Given wind direction (0-360) and list of runways,
    calculates the runway most aligned with the wind.
    """
    if wind_direction_deg is None or not runways:
        return None, ""
        
    best_runway = None
    min_diff = 180.0
    runway_options = []
    
    for rwy in runways:
        rwy_id = rwy.get("id", "")
        if '/' in rwy_id:
            parts = rwy_id.split('/')
            for part in parts:
                runway_options.append(part.strip())
        else:
            runway_options.append(rwy_id.strip())
            
    for r_option in runway_options:
        match = re.match(r'^(\d+)', r_option)
        if match:
            heading = int(match.group(1)) * 10
            diff = abs(wind_direction_deg - heading) % 360
            if diff > 180:
                diff = 360 - diff
            if diff < min_diff:
                min_diff = diff
                best_runway = r_option
                
    if best_runway:
        return best_runway, f"Wind direction ({wind_direction_deg}°) aligns best with Runway {best_runway} (deviation of {min_diff:.0f}°)."
    return None, ""


# ============================================
# DETAILED TAF DECODER (Deterministic)
# ============================================

def _parse_taf_trends(raw_taf: str) -> str:
    """
    Parse TAF trends deterministically using regular expressions.
    Translates raw TAF code lines into conversational, readable summaries.
    """
    if not raw_taf or "Error" in raw_taf or "Exception" in raw_taf or "NOT AVAILABLE" in raw_taf:
        return "TAF forecast data is not available for this station."

    lines = raw_taf.split('\n')
    decoded_trends = []

    for line in lines:
        line = line.strip().upper()
        if not line:
            continue
            
        # Check for change groups
        fm_match = re.search(r'\bFM(\d{2})(\d{2})(\d{2})\b', line)
        tempo_match = re.search(r'\bTEMPO\s+(\d{2})(\d{2})/(\d{2})(\d{2})\b', line)
        becmg_match = re.search(r'\bBECMG\s+(\d{2})(\d{2})/(\d{2})(\d{2})\b', line)
        prob_match = re.search(r'\bPROB(\d{2})\s+(\d{2})(\d{2})/(\d{2})(\d{2})\b', line)
        
        time_prefix = ""
        trend_desc = ""
        
        if fm_match:
            day, hour, minute = fm_match.groups()
            time_prefix = f"From the {day}th day at {hour}:{minute} Zulu: "
            trend_desc = "weather conditions will transition to: "
        elif tempo_match:
            day_start, hour_start, day_end, hour_end = tempo_match.groups()
            time_prefix = f"Temporarily between the {day_start}th at {hour_start}:00 Zulu and {day_end}th at {hour_end}:00 Zulu: "
            trend_desc = "expect occasional fluctuations containing: "
        elif becmg_match:
            day_start, hour_start, day_end, hour_end = becmg_match.groups()
            time_prefix = f"Gradually becoming between the {day_start}th at {hour_start}:00 Zulu and {day_end}th at {hour_end}:00 Zulu: "
            trend_desc = "conditions will evolve to include: "
        elif prob_match:
            prob, day_start, hour_start, day_end, hour_end = prob_match.groups()
            time_prefix = f"There is a {prob}% probability between the {day_start}th at {hour_start}:00 Zulu and {day_end}th at {hour_end}:00 Zulu: "
            trend_desc = "for temporary conditions of: "
            
        if not time_prefix:
            if line.startswith(raw_taf.split()[0]):  # Base forecast line
                time_prefix = "Base Forecast: "
            else:
                continue

        # Parse conditions in this trend line
        conditions = []
        
        # Wind in trend line
        wind_m = re.search(r'\b(\d{3})(\d{2,3})(G(\d{2,3}))?KT\b', line)
        if wind_m:
            dir_w, speed_w, _, gust_w = wind_m.groups()
            wind_str = f"wind from {dir_w}° at {speed_w} knots"
            if gust_w:
                wind_str += f" gusting to {gust_w} knots"
            conditions.append(wind_str)
            
        # Visibility in trend line
        sm_m = re.search(r'\b([PM])?(\d+(?:\s+\d+/\d+)?|\d+/\d+)\s*SM\b', line)
        if sm_m:
            prefix_code, val = sm_m.groups()
            prefix = "greater than " if prefix_code == 'P' else "less than " if prefix_code == 'M' else ""
            conditions.append(f"visibility {prefix}{val} statute miles")
        else:
            four_digit_m = re.search(r'\b(\d{4})\b', line)
            if four_digit_m:
                val = four_digit_m.group(1)
                if val == "9999":
                    conditions.append("visibility 10 kilometers or greater")
                else:
                    conditions.append(f"visibility {int(val)} meters")

        # Clouds in trend line
        cloud_layer_types = {
            'FEW': 'few clouds',
            'SCT': 'scattered clouds',
            'BKN': 'broken ceiling',
            'OVC': 'overcast ceiling'
        }
        cloud_list = []
        for code, name in cloud_layer_types.items():
            matches = re.findall(rf'{code}(\d{{3}})', line)
            for m in matches:
                cloud_list.append(f"{name} at {int(m) * 100:,} feet")
        if cloud_list:
            conditions.append(", ".join(cloud_list))
        elif 'SKC' in line or 'CLR' in line:
            conditions.append("sky clear")
            
        # Weather phenomena in trend line
        wx_map = {
            'TS': 'thunderstorms',
            'RA': 'rain',
            'DZ': 'drizzle',
            'SN': 'snow',
            'FG': 'fog',
            'BR': 'mist',
            'HZ': 'haze'
        }
        wx_found = []
        for word in line.split():
            clean_word = re.sub(r'^[+-]|VC', '', word)
            for code, name in wx_map.items():
                if clean_word == code or (len(clean_word) > 2 and code in clean_word and not clean_word.endswith('KT') and not clean_word.endswith('SM') and not clean_word.startswith('Q') and not clean_word.startswith('A')):
                    if name not in wx_found:
                        wx_found.append(name)
        if wx_found:
            conditions.append(f"weather: {', '.join(wx_found)}")

        if conditions:
            decoded_trends.append(f"• {time_prefix}{trend_desc}{'; '.join(conditions)}.")
            
    if decoded_trends:
        return "\n".join(decoded_trends)
    return "TAF contains standard weather patterns that remain stable throughout the forecast period."


# ============================================
# CORE REGEX METAR PARSER
# ============================================

def _basic_metar_parse(raw_metar: str, raw_taf: str = None, runways: list = None) -> Dict[str, Any]:
    """
    Basic regex-based METAR parsing.
    Extracts weather features and determines flight categories & recommendations.
    """
    # Station (first 4 letters)
    station_match = re.search(r'^([A-Z]{4})', raw_metar)
    station = station_match.group(1) if station_match else "UNKNOWN"
    
    # Time of observation
    time_match = re.search(r'(\d{2})(\d{4})Z', raw_metar)
    if time_match:
        day = time_match.group(1)
        time_str = time_match.group(2)
        time_of_observation = f"{day}th day of month at {time_str[:2]}:{time_str[2:]} Zulu"
    else:
        time_of_observation = "Unable to determine observation time"
    
    # Wind direction & speed
    wind_direction_deg = None
    wind_match = re.search(r'(\d{3})(\d{2,3})(G(\d{2,3}))?KT', raw_metar)
    if wind_match:
        direction = wind_match.group(1)
        speed = wind_match.group(2)
        gust = wind_match.group(4)
        if direction.isdigit():
            wind_direction_deg = int(direction)
            
        if gust:
            surface_wind = f"Wind from {direction}° at {speed} knots gusting to {gust} knots"
        else:
            surface_wind = f"Wind from {direction}° at {speed} knots"
    else:
        # Check for variable wind
        vrb_match = re.search(r'VRB(\d{2})KT', raw_metar)
        if vrb_match:
            surface_wind = f"Wind variable at {vrb_match.group(1)} knots"
        else:
            surface_wind = "Wind data unavailable"
    
    # Visibility
    sm_match = re.search(r'\b([PM])?(\d+(?:\s+\d+/\d+)?|\d+/\d+)\s*SM\b', raw_metar)
    km_match = re.search(r'\b(\d+)\s*KM\b', raw_metar)
    meter_match = re.search(r'\b(\d{4})(?:NDV|[NSEW]{1,2})?\b', raw_metar)
    
    if 'CAVOK' in raw_metar:
        visibility = "Ceiling and Visibility OK (greater than 10 km)"
    else:
        if sm_match:
            prefix_code = sm_match.group(1)
            value = sm_match.group(2)
            value = re.sub(r'\s+', ' ', value)
            prefix = "greater than " if prefix_code == 'P' else "less than " if prefix_code == 'M' else ""
            unit = "statute mile" if value in ["1", "1/2", "1/4", "3/4", "1/8", "5/8"] else "statute miles"
            visibility = f"{prefix}{value} {unit}"
        elif km_match:
            value = km_match.group(1)
            unit = "kilometer" if value == "1" else "kilometers"
            visibility = f"{value} {unit}"
        elif meter_match:
            value = meter_match.group(1)
            if value == "9999":
                visibility = "10 kilometers or greater"
            else:
                visibility = f"{int(value)} meters"
        else:
            visibility = "Visibility data unclear"
    
    # Append weather conditions to visibility description
    wx_conditions = []
    wx_map = {
        'HZ': 'Haze',
        'BR': 'Mist',
        'FG': 'Fog',
        'RA': 'Rain',
        'DZ': 'Drizzle',
        'SN': 'Snow',
        'TS': 'Thunderstorms'
    }
    for word in raw_metar.split()[2:]:
        clean_word = re.sub(r'^[+-]|VC', '', word)
        for code, name in wx_map.items():
            if clean_word == code or (len(clean_word) > 2 and code in clean_word and not clean_word.endswith('KT') and not clean_word.endswith('SM') and not clean_word.startswith('Q') and not clean_word.startswith('A')):
                if name not in wx_conditions:
                    wx_conditions.append(name)
    if wx_conditions and 'CAVOK' not in raw_metar:
        visibility = f"{visibility} in {', '.join(wx_conditions)}"
    
    # Clouds
    cloud_patterns = {
        'FEW': 'Few clouds',
        'SCT': 'Scattered clouds',
        'BKN': 'Broken ceiling',
        'OVC': 'Overcast ceiling'
    }
    cloud_layers = []
    for code, desc in cloud_patterns.items():
        matches = re.findall(rf'{code}(\d{{3}})', raw_metar)
        for match in matches:
            height = int(match) * 100
            cloud_layers.append(f"{desc} at {height:,} feet")
            
    if cloud_layers:
        clouds = ", ".join(cloud_layers)
    elif 'SKC' in raw_metar or 'CLR' in raw_metar:
        clouds = "Sky clear"
    elif 'CAVOK' in raw_metar:
        clouds = "No significant cloud"
    else:
        clouds = "Cloud data unavailable"
    
    # Temperature/Dewpoint
    temp_match = re.search(r'(M?\d{2})/(M?\d{2})', raw_metar)
    if temp_match:
        temp = temp_match.group(1).replace('M', '-')
        dew = temp_match.group(2).replace('M', '-')
        temperature_dew_point = f"Temperature {temp} degrees Celsius, dew point {dew} degrees Celsius"
    else:
        temperature_dew_point = "Temperature data unavailable"
    
    # Altimeter/QNH
    alt_match = re.search(r'A(\d{4})', raw_metar)
    qnh_match = re.search(r'Q(\d{4})', raw_metar)
    
    if alt_match:
        alt_value = alt_match.group(1)
        qnh = f"Altimeter {alt_value[:2]}.{alt_value[2:]} inches of mercury"
    elif qnh_match:
        qnh_value = qnh_match.group(1)
        qnh = f"QNH {qnh_value} hectopascals"
    else:
        qnh = "Altimeter setting unavailable"
    
    # Parse wind values for calculations
    wind_speed = 0
    wind_gust = 0
    gust_spread = 0
    if wind_match:
        try:
            wind_speed = int(wind_match.group(2))
            if wind_match.group(4):
                wind_gust = int(wind_match.group(4))
                gust_spread = wind_gust - wind_speed
            else:
                wind_gust = wind_speed
        except ValueError:
            pass
    elif vrb_match:
        try:
            wind_speed = int(vrb_match.group(1))
            wind_gust = wind_speed
        except ValueError:
            pass

    # Parse visibility in meters
    visibility_m = 10000.0
    if 'CAVOK' in raw_metar:
        visibility_m = 10000.0
    elif sm_match:
        try:
            val_str = sm_match.group(2)
            val_str = re.sub(r'\s+', ' ', val_str).strip()
            parts = val_str.split(' ')
            total_sm = 0.0
            for p in parts:
                if '/' in p:
                    num, denom = p.split('/')
                    total_sm += float(num) / float(denom)
                else:
                    total_sm += float(p)
            visibility_m = total_sm * 1609.34
        except Exception:
            pass
    elif km_match:
        try:
            visibility_m = float(km_match.group(1)) * 1000.0
        except Exception:
            pass
    elif meter_match:
        try:
            visibility_m = float(meter_match.group(1))
        except Exception:
            pass

    # Parse Cloud Ceiling in Feet
    ceiling_ft = 99999.0
    if 'CAVOK' in raw_metar:
        ceiling_ft = 99999.0
    else:
        bkn_ovc_matches = re.findall(r'(BKN|OVC)(\d{3})', raw_metar)
        if bkn_ovc_matches:
            ceiling_heights = []
            for layer_type, height_code in bkn_ovc_matches:
                try:
                    height_ft = int(height_code) * 100
                    ceiling_heights.append(height_ft)
                except ValueError:
                    pass
            if ceiling_heights:
                ceiling_ft = float(min(ceiling_heights))

    # Check for Significant Weather
    has_sig_wx = False
    sig_wx_codes = ['TS', 'RA', 'DZ', 'SN', 'FG', 'BR', 'HZ', 'FU', 'SQ', 'FC', 'GR', 'GS', 'PL']
    metar_words = raw_metar.split()
    for word in metar_words[2:]:
        clean_word = re.sub(r'^[+-]|VC', '', word)
        for code in sig_wx_codes:
            if clean_word == code or (len(clean_word) > 2 and code in clean_word and not clean_word.endswith('KT') and not clean_word.endswith('SM') and not clean_word.startswith('Q') and not clean_word.startswith('A')):
                has_sig_wx = True
                break

    # Determine Flight Category
    if visibility_m < 1600.0 or ceiling_ft < 500.0:
        flight_category = "LIFR"
    elif visibility_m < 5000.0 or ceiling_ft < 1000.0:
        flight_category = "IFR"
    elif visibility_m >= 8000.0 and ceiling_ft >= 3000.0 and not has_sig_wx:
        flight_category = "VFR"
    else:
        flight_category = "MVFR"

    # Evaluate Student Pilot Personal Minimums
    reasons = []
    
    # Visibility (Min 5 SM / 8000m)
    if visibility_m < 8000.0:
        reasons.append(f"Visibility is below student pilot minimum of 5 SM (measured: {visibility_m/1609.34:.1f} SM / {visibility_m:.0f} meters)")
        
    # Ceiling (Min 3,000 ft)
    if ceiling_ft < 3000.0:
        reasons.append(f"Ceiling is below student pilot minimum of 3,000 feet (measured: {ceiling_ft:,.0f} feet)")
        
    # Wind limit
    if wind_speed > 10:
        reasons.append(f"Sustained wind speed is {wind_speed} knots, which exceeds the student maximum limit of 10 knots")
        
    # Wind gust spread (Max 10 KT spread)
    if gust_spread > 10:
        reasons.append(f"Wind gust spread is {gust_spread} knots (sustained {wind_speed}KT, gusting to {wind_gust}KT), which exceeds the student pilot limit of 10 knots")
        
    # Severe Weather
    has_thunderstorm = 'TS' in raw_metar
    has_convective = 'CB' in raw_metar or 'TCU' in raw_metar
    has_wind_shear = 'WS' in raw_metar
    has_icing = 'FZ' in raw_metar
    
    if has_thunderstorm:
        reasons.append("Thunderstorms are reported at the station (TS)")
    if has_convective:
        reasons.append("Convective cloud cells are reported (CB/TCU)")
    if has_wind_shear:
        reasons.append("Low-level wind shear is reported (WS)")
    if has_icing:
        reasons.append("Potential icing/freezing conditions are reported (FZ)")
        
    # Build Flight Recommendation
    if reasons:
        flight_recommendation = "❌ NO-GO\n\nBreaches of student pilot personal minimums:\n" + "\n".join(f"- {r}" for r in reasons)
    else:
        is_marginal = False
        marginal_reasons = []
        if wind_speed == 10:
            is_marginal = True
            marginal_reasons.append("Wind is exactly at the 10-knot threshold")
        if 3000.0 <= ceiling_ft <= 4000.0:
            is_marginal = True
            marginal_reasons.append(f"Ceiling is marginal ({ceiling_ft:,.0f} feet)")
        if 8000.0 <= visibility_m <= 9000.0:
            is_marginal = True
            marginal_reasons.append(f"Visibility is marginal ({visibility_m/1609.34:.1f} SM)")
            
        if is_marginal:
            flight_recommendation = "⚠️ MARGINAL - CONSULT INSTRUCTOR\n\nMarginal conditions noted:\n" + "\n".join(f"- {mr}" for mr in marginal_reasons)
        else:
            flight_recommendation = "✅ GO\n\nAll conditions are well within novice student pilot personal minimums. Have a great flight!"

    # Forecast trend
    forecast_trend = _parse_taf_trends(raw_taf)
    
    # Active Runway Alignment Calculation
    active_rwy_info = ""
    if runways:
        best_rwy, rwy_details = _calculate_active_runway(wind_direction_deg, runways)
        if rwy_details:
            active_rwy_info = f"\n\nActive Runway Analysis: {rwy_details}"
            
    pertinent_information = (
        "🟢 AUTOMATED WEATHER BRIEFING GENERATED VIA REGEX PARSER.\n"
        "Verify all details with official aviation briefings prior to flight."
        f"{active_rwy_info}"
    )

    basic_analysis = {
        "station": station,
        "time_of_observation": time_of_observation,
        "surface_wind": surface_wind,
        "visibility": visibility,
        "clouds": clouds,
        "temperature_dew_point": temperature_dew_point,
        "qnh": qnh,
        "flight_category": flight_category,
        "forecast_trend": forecast_trend,
        "flight_recommendation": flight_recommendation,
        "pertinent_information": pertinent_information
    }
    
    return {
        "status": "success",
        "provider": "regex_parser",
        "analysis": basic_analysis,
        "ai_available": False
    }


# ============================================
# MAIN ORCHESTRATOR
# ============================================

def analyze_metar_orchestrator(raw_metar: str, raw_taf: str = "", use_cache: bool = True) -> Dict[str, Any]:
    """
    Deterministically decode weather reports via regex parsing.
    Saves and reads results from caching mechanism.
    """
    icao = raw_metar[:4].strip().upper() if len(raw_metar) >= 4 else "UNKN"
    
    # Try cache first
    if use_cache:
        cached = weather_cache.get(icao, raw_metar, max_age_seconds=300)
        if cached:
            response = dict(cached)
            response['from_cache'] = True
            response['cache_info'] = "⚡ Instant result from cache (5 min freshness)"
            return response
            
    # Validate inputs
    if not raw_metar or len(raw_metar.strip()) < 10:
        return {
            "status": "error",
            "error_code": "INVALID_INPUT",
            "message": "METAR data is too short or empty",
            "analysis": None,
            "ai_available": False
        }
    
    raw_metar = raw_metar.strip().upper()
    raw_taf = raw_taf.strip().upper() if raw_taf else "NO TAF AVAILABLE"
    
    # Fetch runways list for calculations
    runways = []
    try:
        from src.fetcher import fetch_airport_info
        airport_info = fetch_airport_info(icao)
        if airport_info and "runways" in airport_info:
            runways = airport_info["runways"]
    except Exception as e:
        print(f"⚠️ Failed to load runways for {icao}: {e}")
        
    print(f"🛫 Running Regex Parser for {icao}...")
    response = _basic_metar_parse(raw_metar, raw_taf, runways)
    response["from_cache"] = False
    
    if use_cache:
        weather_cache.set(icao, raw_metar, response)
        
    return response


# ============================================
# CACHED VERSION
# ============================================

@lru_cache(maxsize=128)
def cached_analyze_metar(metar: str, taf: str = "") -> str:
    """
    Cached version of orchestrator for identical requests.
    Returns JSON string for Streamlit caching compatibility.
    """
    result = analyze_metar_orchestrator(metar, taf)
    return json.dumps(result)


# ============================================
# STATUS CHECKS & UTILITIES
# ============================================

def get_ai_status() -> Dict[str, Any]:
    """Get current status of weather parser (mock AI values)"""
    return {
        "deepseek_configured": False,
        "gemini_configured": False,
        "groq_configured": False,
        "openai_configured": False,
        "deepseek_available": False,
        "gemini_available": False,
        "groq_available": False,
        "openai_available": False,
        "rate_limits": {
            "calls_this_minute": {},
            "seconds_until_reset": 0,
            "providers_available": {
                "gemini": False,
                "groq": False,
                "deepseek": False,
                "openai": False
            }
        },
        "environment_valid": True,
        "cache_stats": weather_cache.get_stats(),
        "queue_stats": {
            "queue_length": 0,
            "currently_processing": 0,
            "total_processed": 0,
            "total_queued": 0,
            "max_concurrent": 5
        }
    }


def format_analysis_for_display(analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format the analysis result for clean Streamlit display.
    Adds display-friendly formatting.
    """
    if analysis_result.get("status") not in ["success", "partial"]:
        return analysis_result
    
    analysis = analysis_result.get("analysis", {})
    
    flight_category = analysis.get("flight_category", "").upper()
    category_emoji = {
        "VFR": "🟢",
        "MVFR": "🔵",
        "IFR": "🟠",
        "LIFR": "🔴"
    }
    analysis["flight_category_display"] = f"{category_emoji.get(flight_category, '⚪')} {flight_category}"
    
    recommendation = analysis.get("flight_recommendation", "").upper()
    if "NO-GO" in recommendation:
        analysis["recommendation_badge"] = "🔴 NO-GO"
    elif "GO" in recommendation and "NO-GO" not in recommendation:
        analysis["recommendation_badge"] = "🟢 GO"
    elif "MARGINAL" in recommendation:
        analysis["recommendation_badge"] = "🟡 MARGINAL"
    else:
        analysis["recommendation_badge"] = "⚪ REVIEW"
    
    return analysis_result


# ============================================
# SELF-TEST FUNCTION
# ============================================

def run_tests():
    """Run comprehensive tests on the agent"""
    print("=" * 60)
    print("AVIATION WEATHER REGEX PARSER - TEST SUITE")
    print("=" * 60)
    
    # Test 1: VFR Conditions
    print("\n📋 TEST 1: VFR Conditions (KJFK)")
    print("-" * 40)
    test_metar_vfr = "KJFK 151251Z 18010KT 10SM FEW025 22/15 A3002"
    test_taf_vfr = "KJFK 151200Z 1513/1618 18010KT P6SM FEW025"
    
    result = analyze_metar_orchestrator(test_metar_vfr, test_taf_vfr)
    print(f"Status: {result['status']}")
    print(f"Provider: {result.get('provider', 'N/A')}")
    if result.get('analysis'):
        analysis = result['analysis']
        print(f"Station: {analysis.get('station')}")
        print(f"Category: {analysis.get('flight_category')}")
        print(f"Recommendation:\n{analysis.get('flight_recommendation')}")
        print(f"Forecast Trends:\n{analysis.get('forecast_trend')}")
        print(f"Pertinent Info:\n{analysis.get('pertinent_information')}")
    
    # Test 2: IFR Conditions
    print("\n📋 TEST 2: IFR Conditions (KORD)")
    print("-" * 40)
    test_metar_ifr = "KORD 151451Z 32015G25KT 2SM -RA BR OVC008 12/11 A2975"
    test_taf_ifr = "KORD 151200Z 1513/1618 32015G25KT 2SM -RA BR OVC008 TEMPO 1514/1518 1SM +RA"
    
    result = analyze_metar_orchestrator(test_metar_ifr, test_taf_ifr)
    print(f"Status: {result['status']}")
    print(f"Provider: {result.get('provider', 'N/A')}")
    if result.get('analysis'):
        analysis = result['analysis']
        print(f"Station: {analysis.get('station')}")
        print(f"Category: {analysis.get('flight_category')}")
        print(f"Recommendation:\n{analysis.get('flight_recommendation')}")
        print(f"Forecast Trends:\n{analysis.get('forecast_trend')}")
        print(f"Pertinent Info:\n{analysis.get('pertinent_information')}")
        
    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()