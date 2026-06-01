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

# Import protection layers
from src.cache_manager import weather_cache
from src.queue_manager import request_queue

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
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")


def validate_environment() -> bool:
    """Ensure all required API keys are present and valid format"""
    missing = []
    
    if not GEMINI_KEY:
        missing.append("GEMINI_API_KEY")
    if not GROQ_KEY:
        missing.append("GROQ_API_KEY")
    if not DEEPSEEK_KEY:
        missing.append("DEEPSEEK_API_KEY")
    
    if missing:
        print(f"⚠️ Missing API keys: {', '.join(missing)}")
        print("   App will run but AI features may be limited.")
        return False
    
    # Validate key formats
    if GEMINI_KEY and not GEMINI_KEY.startswith(("AI", "AB")):
        print("⚠️ Gemini API key format looks unusual (expected starting with 'AI' or 'AB')")
    
    if GROQ_KEY and not GROQ_KEY.startswith("gsk_"):
        print("⚠️ Groq API key format looks unusual (expected starting with 'gsk_')")
        
    if DEEPSEEK_KEY and not DEEPSEEK_KEY.startswith("sk-"):
        print("⚠️ DeepSeek API key format looks unusual (expected starting with 'sk-')")
    
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
        description="Critical ATC takeaways (e.g., 'Wind shear reported on final approach', 'ILS approaches required', 'Runway 01R likely in use', 'TEMPO hazards exist'). STRICT RULE: Never name specific active runway numbers unless they are explicitly present in the provided list of actual runways for the airport. If the list is unavailable, state only general wind alignment (e.g., 'wind favors runways aligned with the southeast') or omit naming active runways entirely."
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
   - Identify likely active runway based on wind direction ONLY if the actual runway list of the airport is provided in the prompt. Match the current wind direction with one of the actual runways.
   - STRICT RUNWAY RULE: If no runway list is provided, or the list is empty/unavailable, you MUST NOT guess, invent, or state specific runway numbers (e.g. do not say "Runway 14 is likely active" just because the wind is 140 degrees). Instead, state only the general wind alignment orientation (e.g., "Wind favors runways aligned with the southeast") or omit naming an active runway entirely.
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
    """Improved rate limiter to prevent hitting API limits"""
    
    def __init__(self):
        self.last_call_time: Dict[str, float] = {}
        self.call_counts: Dict[str, int] = {"gemini": 0, "groq": 0, "deepseek": 0}
        self.window_start = time.time()
        self.window_duration = 60  # 1 minute window
        
        # Stricter limits to stay under free tier caps
        self.max_per_minute = {
            "gemini": 10,     # Free: 15/min, stay under
            "groq": 25,       # Free: 30/min, stay under
            "deepseek": 50    # Paid: generous
        }
        
        # Minimum time between calls (seconds)
        self.min_interval = {
            "gemini": 2.0,    # Wait 2s between Gemini calls
            "groq": 0.5,      # Groq is fast
            "deepseek": 1.0   # DeepSeek moderate
        }
    
    def can_call(self, provider: str) -> bool:
        """Check if we can call this provider"""
        now = time.time()
        
        # Reset window if needed
        if now - self.window_start > self.window_duration:
            self.call_counts = {"gemini": 0, "groq": 0, "deepseek": 0}
            self.window_start = now
        
        # Check minute limit
        if self.call_counts.get(provider, 0) >= self.max_per_minute.get(provider, 10):
            return False
        
        # Check cooldown
        if provider in self.last_call_time:
            elapsed = now - self.last_call_time[provider]
            if elapsed < self.min_interval.get(provider, 1.0):
                return False
        
        return True
    
    def record_call(self, provider: str):
        """Record a successful API call"""
        self.last_call_time[provider] = time.time()
        self.call_counts[provider] = self.call_counts.get(provider, 0) + 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current rate limit stats"""
        now = time.time()
        remaining_seconds = max(0, self.window_duration - (now - self.window_start))
        
        return {
            "calls_this_minute": dict(self.call_counts),
            "seconds_until_reset": int(remaining_seconds),
            "providers_available": {
                "gemini": self.can_call("gemini"),
                "groq": self.can_call("groq"),
                "deepseek": self.can_call("deepseek")
            }
        }


# Global rate limiter instance
rate_limiter = RateLimiter()


# ============================================
# CORE FUNCTIONS - AI Provider Calls
# ============================================

def _call_gemini(combined_content: str) -> Optional[str]:
    """
    Call Gemini API with rate limit handling and retry logic.
    """
    max_retries = 2
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                print(f"   Gemini retry {attempt}/{max_retries}...")
                time.sleep(retry_delay * attempt)
            
            client = genai.Client(api_key=GEMINI_KEY)
            
            response = client.models.generate_content(
                model='gemini-2.5-flash',  # Fastest model
                contents=(
                    f"Analyze this raw airport weather data and populate the comprehensive "
                    f"aviation assistant format. Provide thorough, accurate analysis for every field.\n\n"
                    f"WEATHER DATA:\n{combined_content}"
                ),
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,  # Keep full system instructions for 100% accurate pilot safety checks!
                    response_mime_type="application/json",
                    response_schema=METARAnalysis,
                    temperature=0.0,
                    max_output_tokens=800,  # Limit output size for speed
                ),
            )
            
            if response and response.text:
                return response.text
            
            # Check if blocked
            if response and response.prompt_feedback and response.prompt_feedback.block_reason:
                print(f"⚠️ Gemini blocked: {response.prompt_feedback.block_reason}")
                return None
                
            return None
            
        except APIError as e:
            if e.code == 429:  # Rate limit
                if attempt < max_retries:
                    wait_time = retry_delay * (attempt + 1)
                    print(f"⚠️ Gemini rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print("❌ Gemini rate limit persists after retries")
                    raise Exception("GEMINI_ERROR_429")
            else:
                print(f"⚠️ Gemini API Error {e.code}")
                raise Exception(f"GEMINI_ERROR_{e.code}")
        except Exception as e:
            print(f"⚠️ Gemini error: {str(e)[:100]}")
            raise
    
    return None


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
            max_tokens=3000
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


def _extract_json_bruteforce(content: str) -> Optional[str]:
    """
    Extracts JSON from a string by finding the first '{' and last '}'.
    Very robust against markdown wrapping or conversational prefixes/suffixes.
    """
    if not content:
        return None
    try:
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1 and end > start:
            candidate = content[start:end+1]
            # Validate it's actual JSON
            json.loads(candidate)
            return candidate
    except Exception:
        pass
    return None


def _call_deepseek(combined_content: str) -> Optional[str]:
    """
    Call DeepSeek V3 with JSON-focused prompt and robust JSON fallback handling.
    """
    try:
        from openai import OpenAI
        
        client = OpenAI(
            api_key=DEEPSEEK_KEY,
            base_url="https://api.deepseek.com"
        )
        
        # Much simpler, more direct prompt
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a JSON weather API. Your ONLY job is to output valid JSON. "
                        "Never include markdown formatting, code blocks, or any text outside the JSON object. "
                        "Start your response with { and end with }."
                    )
                },
                {
                    "role": "user",
                    "content": f"""Analyze this aviation weather data and output a JSON object with these EXACT 12 fields:

{combined_content}

Output this JSON structure (replace values with your analysis):
{{
  "station": "4-letter ICAO",
  "time_of_observation": "time in plain English",
  "surface_wind": "wind direction and speed in words",
  "visibility": "visibility in plain English",
  "clouds": "cloud layers in plain English",
  "temperature_dew_point": "temperature and dew point in words",
  "qnh": "altimeter setting in plain English",
  "flight_category": "VFR or MVFR or IFR or LIFR",
  "forecast_trend": "TAF summary or No TAF available",
  "flight_recommendation": "Go or No-Go for student pilot with brief reason",
  "educational_insights": "one teaching point from this weather",
  "pertinent_information": "important ATC operational notes"
}}

CRITICAL RUNWAY RULE: Never name specific active runway numbers (e.g. do not suggest Runway 22) unless they are explicitly present in the provided list of actual runways for the airport. If the list is unavailable, empty, or does not contain such runways, state only the general wind alignment (e.g., 'wind favors runways aligned with the southwest') or omit naming active runways entirely. Match the current wind against the actual runways if they are provided.

CRITICAL: Output ONLY the JSON object. No other text."""
                }
            ],
            temperature=0.0,
            max_tokens=800,
            response_format={"type": "json_object"}  # Native JSON mode
        )
        
        content = response.choices[0].message.content
        
        if not content:
            print("⚠️ DeepSeek returned empty response")
            return None
        
        # Validate it's proper JSON
        try:
            # Try parsing directly
            parsed = json.loads(content)
            print(f"✅ DeepSeek JSON valid: {len(content)} chars")
            return json.dumps(parsed)
        except json.JSONDecodeError:
            # Try cleaning with the bruteforce extractor
            print("⚠️ DeepSeek JSON needs cleaning...")
            cleaned = _extract_json_bruteforce(content)
            if cleaned:
                print("✅ DeepSeek JSON cleaned successfully")
                return cleaned
            else:
                print(f"❌ DeepSeek JSON unparseable: {content[:200]}...")
                return None
        
    except Exception as e:
        print(f"⚠️ DeepSeek error: {str(e)[:150]}")
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

def analyze_metar_orchestrator(raw_metar: str, raw_taf: str = "", use_cache: bool = True) -> Dict[str, Any]:
    """
    Smart orchestrator with dual-tier caching and synchronous queue.
    Same airport + same weather = instant response from cache.
    """
    # Extract ICAO from METAR
    icao = raw_metar[:4].strip().upper() if len(raw_metar) >= 4 else "UNKN"
    
    # Layer 1: Try cache first
    if use_cache:
        cached = weather_cache.get(icao, raw_metar, max_age_seconds=300)
        if cached:
            # Create a copy so we don't mutate memory cache
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
    
    # Fetch airport metadata (runway information)
    runways_text = "No runway information available for this airport."
    try:
        from src.fetcher import fetch_airport_info
        airport_info = fetch_airport_info(icao)
        if airport_info and "runways" in airport_info:
            runways = airport_info["runways"]
            if runways:
                runways_formatted = []
                for rwy in runways:
                    rwy_id = rwy.get("id", "Unknown")
                    rwy_dim = rwy.get("dimension", "Unknown")
                    rwy_surf = rwy.get("surface", "Unknown")
                    runways_formatted.append(f"- Runway {rwy_id} (Dimension: {rwy_dim}, Surface: {rwy_surf})")
                runways_text = "Actual Airport Runways:\n" + "\n".join(runways_formatted)
    except Exception as e:
        print(f"⚠️ Failed to fetch/parse runway info: {e}")
        
    combined_content = f"Raw METAR string: {raw_metar}\nRaw TAF block: {raw_taf}\n\n{runways_text}"
    
    print(f"🛫 Analyzing weather data for {icao}...")
    print(f"   METAR: {raw_metar[:80]}...")
    print(f"   TAF: {raw_taf[:80] if raw_taf else 'None'}...")
    
    # Define providers in order of preference
    providers = [
        {
            "name": "gemini", 
            "key": GEMINI_KEY,
            "callable": _call_gemini,
            "emoji": "🤖"
        },
        {
            "name": "groq",
            "key": GROQ_KEY,
            "callable": _call_groq,
            "emoji": "⚡"
        },
        {
            "name": "deepseek",
            "key": DEEPSEEK_KEY,
            "callable": _call_deepseek,
            "emoji": "🌐"
        }
    ]
    
    # Inner helper to process calls inside the queue
    def _call_providers():
        for provider in providers:
            if not provider["key"]:
                print(f"   {provider['emoji']} {provider['name']}: No API key")
                continue
            
            if not rate_limiter.can_call(provider["name"]):
                print(f"   {provider['emoji']} {provider['name']}: Rate limited")
                continue
            
            try:
                print(f"   {provider['emoji']} Trying {provider['name']}...")
                result = provider["callable"](combined_content)
                rate_limiter.record_call(provider["name"])
                
                if result:
                    print(f"   ✅ {provider['name']} success!")
                    response = _validate_and_parse(result, provider["name"])
                    response["rate_limit_stats"] = rate_limiter.get_stats()
                    
                    # If it's a fallback provider, mark fallback_used
                    if provider["name"] != "gemini":
                        response["fallback_used"] = True
                        
                    return response
                else:
                    print(f"   ⚠️ {provider['name']}: No result")
                    
            except Exception as e:
                print(f"   ❌ {provider['name']} failed: {str(e)[:80]}")
                continue
        return None

    # Layer 3: Synchronous Request Queue
    response = request_queue.process_request(_call_providers)
    
    # If a valid response was returned, cache it and return
    if response and response.get("status") in ["success", "partial"]:
        response['from_cache'] = False
        if use_cache:
            weather_cache.set(icao, raw_metar, response)
        return response
        
    # Ultimate fallback if queue/AI failed or all providers failed
    print("   🔧 All AI failed or queue issue, using basic parser")
    response = _basic_metar_parse(raw_metar, raw_taf)
    response["rate_limit_stats"] = rate_limiter.get_stats()
    response["fallback_used"] = True
    response["warning"] = (
        "⚠️ ALL AI SERVICES CURRENTLY OVERLOADED. "
        "Showing basic automated analysis. "
        "This is NOT a substitute for official weather briefing."
    )
    
    # Cache the fallback result too to prevent queue thrashing
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
# UTILITY FUNCTIONS
# ============================================

def get_ai_status() -> Dict[str, Any]:
    """Get current status of all AI providers"""
    from src.cache_manager import weather_cache
    from src.queue_manager import request_queue
    
    return {
        "deepseek_configured": bool(DEEPSEEK_KEY),
        "gemini_configured": bool(GEMINI_KEY),
        "groq_configured": bool(GROQ_KEY),
        "deepseek_available": bool(DEEPSEEK_KEY and rate_limiter.can_call("deepseek")),
        "gemini_available": bool(GEMINI_KEY and rate_limiter.can_call("gemini")),
        "groq_available": bool(GROQ_KEY and rate_limiter.can_call("groq")),
        "rate_limits": rate_limiter.get_stats(),
        "environment_valid": ENV_OK,
        "cache_stats": weather_cache.get_stats(),
        "queue_stats": request_queue.get_stats()
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
# TEMPORARY DEBUG FUNCTION
# ============================================

def debug_deepseek_response():
    """Test what DeepSeek actually returns"""
    from openai import OpenAI
    
    client = OpenAI(
        api_key=DEEPSEEK_KEY,
        base_url="https://api.deepseek.com"
    )
    
    test_metar = "KJFK 151251Z 18010KT 10SM FEW025 22/15 A3002"
    
    print("\n🔍 TESTING DEEPSEEK RESPONSE FORMAT...")
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": "You are a JSON API. Output ONLY valid JSON. No markdown, no extra text."
                },
                {
                    "role": "user",
                    "content": f"Return JSON analysis for: {test_metar}"
                }
            ],
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        print(f"Raw response type: {type(content)}")
        print(f"Response starts with: {repr(content[:50])}...")
        print(f"Response ends with: ...{repr(content[-50:])}")
        print(f"Contains backticks: {'```' in content}")
        print(f"Contains 'json': {'json' in content[:20].lower()}")
        
        try:
            json.loads(content)
            print("✅ Valid JSON!")
        except Exception as e:
            print(f"❌ Invalid JSON - needs cleaning: {e}")
            print(f"Full response:\n{content}")
            
    except Exception as e:
        print(f"❌ DeepSeek Debug Error: {e}")


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    # Run a quick DeepSeek format check first
    if DEEPSEEK_KEY:
        debug_deepseek_response()
    run_tests()