"""
Aviation Weather AI Agent
Multi-provider AI analysis of METAR/TAF data with robust fallback handling.
Supports Gemini (primary) and Groq (fallback) - both free tiers.
"""

import os
import re
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Import Gemini & Groq Tools
from google import genai
from google.genai import types
from google.genai.errors import APIError
from groq import Groq

# ============================================
# ENVIRONMENT SETUP
# ============================================

# Dynamic local environment loading
base_dir = Path(__file__).resolve().parent.parent
env_path = base_dir / ".env"
load_dotenv(dotenv_path=env_path)

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")


def validate_environment() -> bool:
    """Ensure all required API keys are present and valid format"""
    missing = []
    
    if not GEMINI_KEY:
        missing.append("GEMINI_API_KEY")
    if not GROQ_KEY:
        missing.append("GROQ_API_KEY")
    
    if missing:
        print(f"⚠️ Missing API keys: {', '.join(missing)}")
        print("   App will run but AI features may be limited.")
        return False
    
    # Validate key formats
    if GEMINI_KEY and not GEMINI_KEY.startswith(("AI", "AB")):
        print("⚠️ Gemini API key format looks unusual (expected starting with 'AI' or 'AB')")
    
    if GROQ_KEY and not GROQ_KEY.startswith("gsk_"):
        print("⚠️ Groq API key format looks unusual (expected starting with 'gsk_')")
    
    return True


# Validate on import
ENV_OK = validate_environment()


# ============================================
# PYDANTIC SCHEMA - Comprehensive Analysis
# ============================================

class METARAnalysis(BaseModel):
    """Complete aviation weather analysis schema"""
    
    station: str = Field(
        description="The 4-letter ICAO code of the station"
    )
    
    time_of_observation: str = Field(
        description="The time of observation extracted from the METAR (e.g., '0530 Zulu on the 15th')"
    )
    
    surface_wind: str = Field(
        description="Wind direction and speed, fully decoded into spoken text (e.g., 'Wind from the northwest at 15 knots gusting to 25')"
    )
    
    visibility: str = Field(
        description="Visibility values fully decoded into words (e.g., '10 statute miles' or '2 kilometers')"
    )
    
    clouds: str = Field(
        description="Cloud layers and ceilings fully decoded into spoken text (e.g., 'Scattered clouds at 2,500 feet, broken ceiling at 5,000 feet')"
    )
    
    temperature_dew_point: str = Field(
        description="The air temperature and dew point values written out in full words (e.g., 'Temperature 22 degrees Celsius, dew point 15 degrees Celsius')"
    )
    
    qnh: str = Field(
        description="The altimeter setting / QNH value written out fully (e.g., 'Altimeter 30.02 inches of mercury' or 'QNH 1017 hectopascals')"
    )
    
    flight_category: str = Field(
        description="Strict current operational category: VFR, MVFR, IFR, or LIFR"
    )
    
    forecast_trend: str = Field(
        description="A clear summary of upcoming weather changes based on the TAF (e.g., 'Conditions expected to deteriorate after 1800Z with lowering ceilings and rain')"
    )
    
    flight_recommendation: str = Field(
        description="A strict Go/No-Go safety recommendation specifically tailored for an early student pilot. Must detail wind limits, runway visual ranges, ceiling constraints, and explain WHY."
    )
    
    educational_insights: str = Field(
        description="A comprehensive, highly structured, line-by-line breakdown explaining all parts of the METAR and especially the TAF codes in detailed, sequential sections. List raw code segments and explain every element in detail. Conclude with a checkride Q&A."
    )
    
    pertinent_information: str = Field(
        description="Critical ATC takeaways (e.g., 'Wind shear reported on final approach', 'ILS approaches required', 'Runway 27 likely in use', 'TEMPO hazards exist')"
    )


# ============================================
# SYSTEM INSTRUCTION - AI Prompt Engineering
# ============================================

SYSTEM_INSTRUCTION = """
You are an expert Air Traffic Control assistant and Senior Flight Instructor with 20 years of experience. 
Your job is to decode raw METAR and TAF data into professional aviation phraseology and evaluate operational impacts.

═══════════════════════════════════════
CRITICAL RULES - MUST FOLLOW EXACTLY
═══════════════════════════════════════

1. IFR CLASSIFICATION (ABSOLUTE RULES):
   - If visibility < 5000 meters OR ceiling < 1000 feet → Classify as IFR
   - If visibility < 1600 meters OR ceiling < 500 feet → Classify as LIFR
   - If BOTH conditions exist, LIFR takes priority
   - If visibility is 5000+ meters AND ceiling 1000+ feet → MVFR
   - If visibility is 8000+ meters AND ceiling 3000+ feet AND no significant weather → VFR

2. STUDENT PILOT FLIGHT RECOMMENDATION (STRICT MINIMUMS):
   Assume a novice student pilot (pre-solo, less than 20 hours) with these personal minimums:
   - MAX crosswind component: 10 knots
   - MIN visibility: 5 statute miles (8000 meters)
   - MIN ceiling: 3000 feet above ground level
   - MAX gust spread: 10 knots (difference between sustained and gust)
   - ABSOLUTELY NO: thunderstorms, convective activity, icing conditions, or LLWS (low-level wind shear)
   - If ANY minimum is breached → Return "NO-GO" with specific reason
   - If ALL minimums met → Return "GO" with confidence level
   - If borderline → Return "MARGINAL - CONSULT INSTRUCTOR"

3. EDUCATIONAL INSIGHTS (TEACHING VALUE):
   - You MUST generate a comprehensive, highly structured "Line-by-Line Breakdown" of the weather codes (especially the TAF blocks, but also the METAR).
   - Begin with the bold title: "### 📝 The Line-by-Line Breakdown"
   - Go through each logical period/line of the weather reports, numbering them sequentially (e.g., "1. The Base Indicator & Validity Period", "2. The Thunderstorm Hazard (TEMPO)", "3. The Evening Transition (BECMG)", etc.).
   - Under each section, output the raw code block on a separate line under a bold "Plaintext" header.
   - For every individual code element inside that raw block (e.g., station name, date/time, wind speed/direction, gusts, visibility, cloud layers, and weather hazards), list the element followed by a detailed, easy-to-understand breakdown in plain English (e.g., "TAF VIDP: Terminal Aerodrome Forecast for...", "010500Z: Issued on...", "10010KT: Base winds are...", "6000: Ground visibility is forecast at...").
   - Follow this exact formatting style for each section:
     X. [Section Title]
     **Plaintext**
     [raw code block]
     [code element 1]: [detailed plaintext explanation]
     [code element 2]: [detailed plaintext explanation]
     ...
   - Do not omit any elements; ensure every single part of the reported code is explained line by line.
   - Conclude at the very end of the insights with ONE mock private pilot checkride oral exam question and answer based on this SPECIFIC weather. Format it clearly as:
     "**Checkride Q:** [challenging checkride question about a code or safety implication in this report]\n**Checkride A:** [complete, accurate checkride answer]"

4. PERTINENT ATC INFORMATION:
   - Note anything requiring Air Traffic Controller attention
   - Identify likely active runway based on wind direction
   - Flag any SPECI (special weather report) conditions
   - Note if instrument approaches (ILS, RNAV) would be required
   - Identify any NOTAM-worthy conditions

5. TEXT FORMATTING STANDARDS:
    - Use conversational, professional aviation language
    - Write out all units: "knots" NOT "KT", "miles" NOT "SM", "feet" NOT "FT"
    - Write temperatures as "25 degrees Celsius" NOT "25°C"
    - Write altimeter/QNH settings exactly in their reported units: use "Altimeter 30.02 inches of mercury" for North American 'A' settings, and "QNH 1013 hectopascals" for international 'Q' settings (do NOT convert between units or map hectopascals into inches of mercury).
    - Never use raw METAR codes in output (except in educational section where explaining them)
    - Be concise but thorough - aim for 3-5 sentences per field

6. MISSING DATA HANDLING:
   - If TAF is not available or incomplete, state "TAF data unavailable for this station" in forecast_trend
   - Never fabricate weather data
   - If uncertain about any value, note the uncertainty
"""


# ============================================
# RATE LIMITER - Protect Free API Tiers (Cross-Process & Sliding Window)
# ============================================

class RateLimiter:
    """Persistent, cross-process sliding-window rate limiter using a local JSON file"""
    
    def __init__(self, cache_file: Optional[Path] = None):
        if cache_file is None:
            self.cache_file = Path(__file__).resolve().parent / "rate_limit_state.json"
        else:
            self.cache_file = cache_file
            
        self.limits = {
            "gemini": {"max_rpm": 14, "min_spacing": 4.0}, # 4.0s spacing ensures max 15 RPM globally
            "groq": {"max_rpm": 28, "min_spacing": 2.0}    # 2.0s spacing ensures max 30 RPM globally
        }
        self._init_cache()

    def _init_cache(self):
        """Ensure the rate limit cache file exists and is valid"""
        if not self.cache_file.exists():
            self._write_state({"gemini": [], "groq": []})

    def _read_state(self) -> Dict[str, list]:
        try:
            if self.cache_file.exists():
                return json.loads(self.cache_file.read_text(encoding='utf-8'))
        except Exception:
            pass
        return {"gemini": [], "groq": []}

    def _write_state(self, state: Dict[str, list]):
        try:
            self.cache_file.write_text(json.dumps(state), encoding='utf-8')
        except Exception as e:
            # Silently fallback if disk writes are temporarily locked
            pass

    def can_call(self, provider: str) -> bool:
        """Check if provider can be called based on shared cross-process limits"""
        if provider not in self.limits:
            return True
            
        now = time.time()
        state = self._read_state()
        timestamps = state.get(provider, [])
        
        # Clean up timestamps older than 60 seconds (sliding window)
        timestamps = [t for t in timestamps if now - t < 60]
        
        # Check RPM limit
        if len(timestamps) >= self.limits[provider]["max_rpm"]:
            return False
            
        # Check minimum spacing between calls
        if timestamps:
            last_call = timestamps[-1]
            if now - last_call < self.limits[provider]["min_spacing"]:
                return False
                
        return True

    def record_call(self, provider: str):
        """Record that a call was made across all processes"""
        if provider not in self.limits:
            return
            
        now = time.time()
        state = self._read_state()
        timestamps = state.get(provider, [])
        
        # Clean up old timestamps and append new call
        timestamps = [t for t in timestamps if now - t < 60]
        timestamps.append(now)
        state[provider] = timestamps
        
        self._write_state(state)

    def get_stats(self) -> Dict[str, Any]:
        """Get current rate limit statistics"""
        now = time.time()
        state = self._read_state()
        stats = {}
        providers_avail = {}
        
        for provider in ["gemini", "groq"]:
            timestamps = state.get(provider, [])
            timestamps = [t for t in timestamps if now - t < 60]
            stats[provider] = len(timestamps)
            
            # Check availability
            avail = True
            if len(timestamps) >= self.limits[provider]["max_rpm"]:
                avail = False
            elif timestamps and now - timestamps[-1] < self.limits[provider]["min_spacing"]:
                avail = False
            providers_avail[provider] = avail
            
        return {
            "calls_this_minute": stats,
            "seconds_until_reset": max(0, int(60 - (now - state.get("gemini", [now])[0]))) if state.get("gemini") else 0,
            "providers_available": providers_avail
        }


# Global rate limiter instance
rate_limiter = RateLimiter()


# ============================================
# CORE FUNCTIONS - AI Provider Calls
# ============================================

def _call_gemini(combined_content: str) -> Optional[str]:
    """
    Call Gemini API for METAR analysis.
    Returns JSON string on success, None on failure.
    """
    try:
        client = genai.Client(api_key=GEMINI_KEY)
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=(
                f"Analyze this raw airport weather data and populate the comprehensive "
                f"aviation assistant format. Provide thorough, accurate analysis for every field.\n\n"
                f"WEATHER DATA:\n{combined_content}"
            ),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=METARAnalysis,
                temperature=0.0,
                max_output_tokens=8192,
                top_p=0.95
            ),
        )
        
        # Check for valid response
        if response and response.text:
            # Check if content was blocked
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                print(f"⚠️ Gemini content blocked: {response.prompt_feedback.block_reason}")
                return None
            
            return response.text
        
        # Handle safety filters
        if response and hasattr(response, 'candidates') and not response.candidates:
            print("⚠️ Gemini returned no candidates (possible safety filter)")
            return None
        
        print("⚠️ Gemini returned empty response")
        return None
        
    except APIError as e:
        error_messages = {
            400: "Bad request - check METAR format",
            401: "Invalid API key",
            429: "Rate limit exceeded",
            500: "Gemini server error",
            503: "Gemini service unavailable"
        }
        msg = error_messages.get(e.code, f"Unknown error")
        print(f"⚠️ Gemini API Error {e.code}: {msg}")
        raise Exception(f"GEMINI_ERROR_{e.code}")
        
    except Exception as e:
        print(f"⚠️ Gemini unexpected error: {str(e)[:150]}")
        raise


def _call_groq(combined_content: str) -> Optional[str]:
    """
    Call Groq API for METAR analysis (fallback).
    Returns JSON string on success, None on failure.
    """
    try:
        client = Groq(api_key=GROQ_KEY)
        
        # Convert Pydantic schema to detailed JSON instructions
        schema_fields = list(METARAnalysis.model_fields.keys())
        schema_description = {}
        for field in schema_fields:
            schema_description[field] = METARAnalysis.model_fields[field].description
        
        prompt_with_schema = (
            f"You are an expert aviation weather analyst.\n\n"
            f"Analyze this weather data thoroughly:\n{combined_content}\n\n"
            f"You MUST return a valid JSON object with EXACTLY these fields:\n"
            f"{json.dumps(schema_description, indent=2)}\n\n"
            f"CRITICAL SYSTEM RULES:\n{SYSTEM_INSTRUCTION}\n\n"
            f"Remember: Return ONLY the JSON object, no additional text."
        )
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a JSON-only aviation weather analyst. Always return complete, valid JSON with all required fields. Never include text outside the JSON object."
                },
                {
                    "role": "user",
                    "content": prompt_with_schema
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=1500
        )
        
        content = response.choices[0].message.content
        
        # Validate it's actual JSON
        if content:
            try:
                json.loads(content)
                return content
            except json.JSONDecodeError:
                # Try to extract JSON if Groq added extra text
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    extracted = json_match.group()
                    try:
                        json.loads(extracted)
                        print("⚠️ Extracted JSON from Groq response (had extra text)")
                        return extracted
                    except:
                        pass
                print("⚠️ Groq returned invalid JSON")
                return None
        
        return None
        
    except Exception as e:
        print(f"⚠️ Groq error: {str(e)[:150]}")
        raise


# ============================================
# BASIC METAR PARSER - Ultimate Fallback
# ============================================

def _basic_metar_parse(raw_metar: str, raw_taf: str = None) -> Dict[str, Any]:
    """
    Basic regex-based METAR parsing when all AI services fail.
    Extracts what we can without AI.
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
    
    # Wind
    wind_match = re.search(r'(\d{3})(\d{2,3})(G(\d{2,3}))?KT', raw_metar)
    if wind_match:
        direction = wind_match.group(1)
        speed = wind_match.group(2)
        gust = wind_match.group(4)
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
    vis_match = re.search(r'(\d+)(SM|KM)', raw_metar)
    if vis_match:
        value = vis_match.group(1)
        unit = "statute miles" if vis_match.group(2) == "SM" else "kilometers"
        visibility = f"{value} {unit}"
    elif re.search(r'9999', raw_metar):
        visibility = "10 kilometers or greater"
    elif re.search(r'CAVOK', raw_metar):
        visibility = "Ceiling and Visibility OK (greater than 10 km)"
    else:
        visibility = "Visibility data unclear"
    
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
    
    # Flight category
    flight_category = "UNABLE TO DETERMINE - Exercise caution"
    
    # Forecast trend
    if raw_taf and len(raw_taf) > 20:
        forecast_trend = f"TAF available but could not be analyzed. First portion: {raw_taf[:150]}..."
    else:
        forecast_trend = "TAF data unavailable for this station"
    
    # Build response
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
        "flight_recommendation": (
            "⚠️ AUTOMATED FALLBACK - AI ANALYSIS UNAVAILABLE. "
            "Cannot provide safety recommendation without AI interpretation. "
            "Student pilots must consult a certified flight instructor or official weather briefing."
        ),
        "educational_insights": (
            "⚠️ TEACHING MODE OFFLINE - AI services currently unavailable. "
            "Review the raw METAR manually with your instructor. "
            "Key elements to discuss: wind speed/direction, visibility, cloud ceilings, and any significant weather codes present."
        ),
        "pertinent_information": (
            "⚠️ BASIC AUTOMATED ANALYSIS ONLY. "
            "Verify all information with official aviation weather sources before flight. "
            "ATC-related insights unavailable without AI processing."
        )
    }
    
    return {
        "status": "fallback_basic",
        "provider": "regex_parser",
        "analysis": basic_analysis,
        "warning": "⚠️ AI SERVICES UNAVAILABLE - Using basic automated parsing. VERIFY WITH OFFICIAL SOURCES.",
        "ai_available": False
    }


# ============================================
# RESPONSE VALIDATION
# ============================================

def _validate_and_parse(ai_response: str, provider: str) -> Dict[str, Any]:
    """
    Parse and validate AI JSON response against Pydantic schema.
    Returns structured dict with status metadata.
    """
    try:
        # Parse JSON
        data = json.loads(ai_response)
        
        # Validate against Pydantic schema
        validated = METARAnalysis(**data)
        
        # Return success response
        return {
            "status": "success",
            "provider": provider,
            "analysis": validated.model_dump(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "ai_available": True
        }
        
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON parse error from {provider}: {str(e)[:100]}")
        print(f"   Raw response snippet: {repr(ai_response[:200])}...")
        return {
            "status": "error",
            "error_code": "JSON_PARSE_ERROR",
            "message": f"AI returned invalid JSON format: {str(e)[:100]}",
            "provider": provider,
            "analysis": None,
            "ai_available": False
        }
        
    except Exception as e:
        print(f"⚠️ Schema validation error from {provider}: {str(e)[:100]}")
        
        # Try to return partial data
        try:
            partial_data = json.loads(ai_response)
            return {
                "status": "partial",
                "provider": provider,
                "analysis": partial_data,
                "warning": f"Some fields may be missing or invalid: {str(e)[:100]}",
                "ai_available": True
            }
        except:
            return {
                "status": "error",
                "error_code": "VALIDATION_ERROR",
                "message": f"Failed to validate AI response: {str(e)[:100]}",
                "provider": provider,
                "analysis": None,
                "ai_available": False
            }


# ============================================
# MAIN ORCHESTRATOR
# ============================================

def analyze_metar_orchestrator(raw_metar: str, raw_taf: str = "") -> Dict[str, Any]:
    """
    Main orchestrator for METAR/TAF analysis.
    
    Attempts Gemini first, falls back to Groq, then basic parsing.
    Returns structured dict with analysis and metadata.
    
    Args:
        raw_metar: Raw METAR string
        raw_taf: Raw TAF string (optional)
    
    Returns:
        Dict with keys: status, provider, analysis, timestamp, ai_available
    """
    # Validate inputs
    if not raw_metar or len(raw_metar.strip()) < 10:
        return {
            "status": "error",
            "error_code": "INVALID_INPUT",
            "message": "METAR data is too short or empty",
            "analysis": None,
            "ai_available": False
        }
    
    # Clean inputs
    raw_metar = raw_metar.strip().upper()
    raw_taf = raw_taf.strip().upper() if raw_taf else "NO TAF AVAILABLE"
    
    # Prepare combined content
    combined_content = f"Raw METAR string: {raw_metar}\nRaw TAF block: {raw_taf}"
    
    print(f"🛫 Analyzing weather data...")
    print(f"   METAR: {raw_metar[:80]}...")
    print(f"   TAF: {raw_taf[:80] if raw_taf else 'None'}...")
    
    # === ATTEMPT 1: Gemini ===
    if GEMINI_KEY and rate_limiter.can_call("gemini"):
        try:
            print("🤖 Attempting analysis with Gemini 2.5 Flash...")
            result = _call_gemini(combined_content)
            rate_limiter.record_call("gemini")
            
            if result:
                print("✅ Gemini analysis successful")
                response = _validate_and_parse(result, "gemini")
                response["rate_limit_stats"] = rate_limiter.get_stats()
                return response
            else:
                print("⚠️ Gemini returned no usable result")
                
        except Exception as e:
            error_msg = str(e)[:100]
            print(f"⚠️ Gemini failed: {error_msg}")
            # Don't record failed calls
    else:
        if not GEMINI_KEY:
            print("⚠️ Gemini API key not configured")
        elif not rate_limiter.can_call("gemini"):
            print("⚠️ Gemini rate limit reached, skipping to fallback")
    
    # === ATTEMPT 2: Groq ===
    if GROQ_KEY and rate_limiter.can_call("groq"):
        try:
            print("🔄 Attempting analysis with Groq (Llama 3.3)...")
            result = _call_groq(combined_content)
            rate_limiter.record_call("groq")
            
            if result:
                print("✅ Groq analysis successful")
                response = _validate_and_parse(result, "groq")
                response["rate_limit_stats"] = rate_limiter.get_stats()
                response["fallback_used"] = True
                return response
            else:
                print("⚠️ Groq returned no usable result")
                
        except Exception as e:
            error_msg = str(e)[:100]
            print(f"⚠️ Groq failed: {error_msg}")
    else:
        if not GROQ_KEY:
            print("⚠️ Groq API key not configured")
        elif not rate_limiter.can_call("groq"):
            print("⚠️ Groq rate limit reached")
    
    # === ATTEMPT 3: Ultimate Fallback ===
    print("⚠️ All AI providers unavailable. Using basic regex parser...")
    response = _basic_metar_parse(raw_metar, raw_taf)
    response["rate_limit_stats"] = rate_limiter.get_stats()
    response["fallback_used"] = True
    response["warning"] = (
        "⚠️ ALL AI SERVICES CURRENTLY UNAVAILABLE. "
        "Showing basic automated analysis. "
        "This is NOT a substitute for official weather briefing."
    )
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
# UTILITY FUNCTIONS
# ============================================

def get_ai_status() -> Dict[str, Any]:
    """Get current status of all AI providers"""
    return {
        "gemini_configured": bool(GEMINI_KEY),
        "groq_configured": bool(GROQ_KEY),
        "gemini_available": bool(GEMINI_KEY and rate_limiter.can_call("gemini")),
        "groq_available": bool(GROQ_KEY and rate_limiter.can_call("groq")),
        "rate_limits": rate_limiter.get_stats(),
        "environment_valid": ENV_OK
    }


def format_analysis_for_display(analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format the analysis result for clean Streamlit display.
    Adds display-friendly formatting.
    """
    if analysis_result.get("status") != "success" and analysis_result.get("status") != "partial":
        return analysis_result
    
    analysis = analysis_result.get("analysis", {})
    
    # Add emoji indicators
    flight_category = analysis.get("flight_category", "").upper()
    category_emoji = {
        "VFR": "🟢",
        "MVFR": "🔵",
        "IFR": "🟠",
        "LIFR": "🔴"
    }
    analysis["flight_category_display"] = f"{category_emoji.get(flight_category, '⚪')} {flight_category}"
    
    # Parse Go/No-Go from recommendation
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
    print("AVIATION WEATHER AI AGENT - TEST SUITE")
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
        print(f"Recommendation: {analysis.get('flight_recommendation', '')[:150]}...")
    
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
        print(f"Recommendation: {analysis.get('flight_recommendation', '')[:150]}...")
    
    # Test 3: No TAF Available
    print("\n📋 TEST 3: Missing TAF (EGLL)")
    print("-" * 40)
    test_metar_no_taf = "EGLL 151220Z 22005KT 9999 BKN040 18/10 Q1020 NOSIG"
    
    result = analyze_metar_orchestrator(test_metar_no_taf, "")
    print(f"Status: {result['status']}")
    print(f"Provider: {result.get('provider', 'N/A')}")
    if result.get('analysis'):
        print(f"Forecast: {result['analysis'].get('forecast_trend', '')[:100]}...")
    
    # Test 4: Edge Case - Very Short METAR
    print("\n📋 TEST 4: Invalid METAR (Edge Case)")
    print("-" * 40)
    test_bad_metar = "KJFK"
    
    result = analyze_metar_orchestrator(test_bad_metar, "")
    print(f"Status: {result['status']}")
    print(f"Error: {result.get('message', 'N/A')}")
    
    # AI Status
    print("\n📊 AI PROVIDER STATUS")
    print("-" * 40)
    status = get_ai_status()
    print(json.dumps(status, indent=2))
    
    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETE")
    print("=" * 60)


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    run_tests()