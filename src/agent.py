"""
Aviation Weather AI Agent
Multi-provider AI analysis of METAR/TAF data with robust fallback handling.
Supports Gemini, Groq, OpenAI, and DeepSeek.
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

# Import AI Provider Tools
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
OPENAI_KEY = os.getenv("OPENAI_API_KEY")


def validate_environment() -> bool:
    """Ensure all required API keys are present and valid format"""
    missing = []
    
    if not GEMINI_KEY:
        missing.append("GEMINI_API_KEY")
    if not GROQ_KEY:
        missing.append("GROQ_API_KEY")
    if not DEEPSEEK_KEY:
        missing.append("DEEPSEEK_API_KEY")
    if not OPENAI_KEY:
        missing.append("OPENAI_API_KEY")
    
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
        
    if OPENAI_KEY and not OPENAI_KEY.startswith("sk-"):
        print("⚠️ OpenAI API key format looks unusual (expected starting with 'sk-')")
    
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

3. PERTINENT ATC INFORMATION:
   - Note anything requiring Air Traffic Controller attention
   - Identify likely active runway based on wind direction ONLY if the actual runway list of the airport is provided in the prompt. Match the current wind direction with one of the actual runways.
   - STRICT RUNWAY RULE: If no runway list is provided, or the list is empty/unavailable, you MUST NOT guess, invent, or state specific runway numbers (e.g. do not say "Runway 14 is likely active" just because the wind is 140 degrees). Instead, state only the general wind alignment orientation (e.g., "Wind favors runways aligned with the southeast") or omit naming an active runway entirely.
   - Flag any SPECI (special weather report) conditions
   - Note if instrument approaches (ILS, RNAV) would be required
   - Identify any NOTAM-worthy conditions

4. TEXT FORMATTING STANDARDS:
    - Use conversational, professional aviation language
    - Write out all units: "knots" NOT "KT", "miles" NOT "SM", "feet" NOT "FT"
    - Write temperatures as "25 degrees Celsius" NOT "25°C"
    - Write altimeter/QNH settings exactly in their reported units: use "Altimeter 30.02 inches of mercury" for North American 'A' settings, and "QNH 1013 hectopascals" for international 'Q' settings (do NOT convert between units or map hectopascals into inches of mercury).
    - Never use raw METAR codes in output
    - Be concise but thorough - aim for 3-5 sentences per field

5. MISSING DATA HANDLING:
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
        self.call_counts: Dict[str, int] = {"gemini": 0, "groq": 0, "deepseek": 0, "openai": 0}
        self.window_start = time.time()
        self.window_duration = 60  # 1 minute window
        
        # Stricter limits to stay under free tier caps
        self.max_per_minute = {
            "gemini": 10,     # Free: 15/min, stay under
            "groq": 25,       # Free: 30/min, stay under
            "deepseek": 50,   # Paid: generous
            "openai": 30      # Paid/Tier: generous
        }
        
        # Minimum time between calls (seconds)
        self.min_interval = {
            "gemini": 2.0,    # Wait 2s between Gemini calls
            "groq": 0.5,      # Groq is fast
            "deepseek": 1.0,  # DeepSeek moderate
            "openai": 0.5     # OpenAI is fast
        }
    
    def can_call(self, provider: str) -> bool:
        """Check if we can call this provider"""
        now = time.time()
        
        # Reset window if needed
        if now - self.window_start > self.window_duration:
            self.call_counts = {"gemini": 0, "groq": 0, "deepseek": 0, "openai": 0}
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
                "deepseek": self.can_call("deepseek"),
                "openai": self.can_call("openai")
            }
        }


# Global rate limiter instance
rate_limiter = RateLimiter()


# ============================================
# CORE FUNCTIONS - AI Provider Calls
# ============================================

def _call_gemini(combined_content: str) -> Optional[str]:
    """
    Call Gemini API via direct REST request to avoid Windows SDK socket timeouts.
    """
    import requests
    
    max_retries = 2
    retry_delay = 2  # seconds
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # Convert Pydantic schema to detailed JSON instructions for unconstrained JSON mode
    schema_fields = list(METARAnalysis.model_fields.keys())
    schema_description = {}
    for field in schema_fields:
        schema_description[field] = METARAnalysis.model_fields[field].description
    
    prompt_with_schema = (
        f"Analyze this aviation weather data and output a JSON object with these EXACT 12 fields:\n\n"
        f"{combined_content}\n\n"
        f"Output this JSON structure (replace values with your analysis):\n"
        f"{json.dumps(schema_description, indent=2)}\n\n"
        f"Remember: Output ONLY the JSON object. Do NOT use markdown code blocks or any other wrapping. Start your response with {{ and end with }}."
    )
    
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt_with_schema
                    }
                ]
            }
        ],
        "systemInstruction": {
            "parts": [
                {
                    "text": SYSTEM_INSTRUCTION
                }
            ]
        },
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.0,
            "maxOutputTokens": 2000
        }
    }
    
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                print(f"   Gemini retry {attempt}/{max_retries}...")
                time.sleep(retry_delay * attempt)
                
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                # Check for prompt block
                if "candidates" not in data or not data["candidates"]:
                    print(f"⚠️ Gemini REST returned no candidates: {data}")
                    return None
                    
                candidate = data["candidates"][0]
                if "content" not in candidate or "parts" not in candidate["content"]:
                    # Check if blocked
                    if "finishReason" in candidate and candidate["finishReason"] == "SAFETY":
                        print(f"⚠️ Gemini blocked: {candidate.get('finishReason')}")
                    return None
                    
                text = candidate["content"]["parts"][0]["text"]
                return text
                
            elif response.status_code == 429:
                print("⚠️ Gemini rate limited (HTTP 429)")
                if attempt < max_retries:
                    wait_time = retry_delay * (attempt + 1)
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception("GEMINI_ERROR_429")
            else:
                print(f"⚠️ Gemini API returned HTTP {response.status_code}: {response.text[:200]}")
                raise Exception(f"GEMINI_ERROR_{response.status_code}")
                
        except Exception as e:
            if attempt == max_retries:
                print(f"⚠️ Gemini REST failure: {str(e)[:150]}")
                raise
            continue
            
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
            content = _clean_json_string(content)
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


def _call_openai(combined_content: str) -> Optional[str]:
    """
    Call OpenAI API for METAR analysis (fallback).
    Returns JSON string on success, None on failure.
    """
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=OPENAI_KEY)
        
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
        
        # Try utilizing OpenAI's native Structured Outputs for 100% schema enforcement
        try:
            response = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a JSON-only aviation weather analyst. Always return complete, valid JSON with all required fields matching the schema. Never include text outside the JSON object."
                    },
                    {
                        "role": "user",
                        "content": f"Analyze this weather data:\n{combined_content}\n\nCRITICAL SYSTEM RULES:\n{SYSTEM_INSTRUCTION}"
                    }
                ],
                response_format=METARAnalysis,
                temperature=0.0
            )
            content = response.choices[0].message.content
        except AttributeError:
            # Fallback if the local openai library version doesn't support .parse
            response = client.chat.completions.create(
                model="gpt-4o-mini",
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
                max_tokens=2000
            )
            content = response.choices[0].message.content
        
        if content:
            content = _clean_json_string(content)
            try:
                json.loads(content)
                return content
            except json.JSONDecodeError:
                # Try to extract JSON if OpenAI added extra text
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    extracted = json_match.group()
                    try:
                        json.loads(extracted)
                        print("⚠️ Extracted JSON from OpenAI response (had extra text)")
                        return extracted
                    except:
                        pass
                print("⚠️ OpenAI returned invalid JSON")
                return None
        
        return None
        
    except Exception as e:
        print(f"⚠️ OpenAI error: {str(e)[:150]}")
        raise


def _clean_json_string(s: str) -> str:
    """
    Clean JSON string by replacing non-breaking spaces (\xa0) and
    zero-width spaces (\u200b) with standard spacing.
    """
    if not s:
        return ""
    return s.replace('\xa0', ' ').replace('\u200b', '').strip()


def _extract_json_bruteforce(content: str) -> Optional[str]:
    """
    Extracts JSON from a string by finding the first '{' and last '}'.
    Very robust against markdown wrapping or conversational prefixes/suffixes.
    """
    if not content:
        return None
    content = _clean_json_string(content)
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
  "pertinent_information": "important ATC operational notes"
}}

CRITICAL RUNWAY RULE: Never name specific active runway numbers (e.g. do not suggest Runway 22) unless they are explicitly present in the provided list of actual runways for the airport. If the list is unavailable, empty, or does not contain such runways, state only the general wind alignment (e.g., 'wind favors runways aligned with the southwest') or omit naming active runways entirely. Match the current wind against the actual runways if they are provided.

CRITICAL: Output ONLY the JSON object. No other text."""
                }
            ],
            temperature=0.0,
            max_tokens=2000,
            response_format={"type": "json_object"}  # Native JSON mode
        )
        
        content = response.choices[0].message.content
        
        if not content:
            print("⚠️ DeepSeek returned empty response")
            return None
        
        content = _clean_json_string(content)
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
    if 'CAVOK' in raw_metar:
        visibility = "Ceiling and Visibility OK (greater than 10 km)"
    else:
        # 1. North American visibility with SM (e.g., 10SM, P6SM, M1/4SM, 1 1/2SM, 1 1/2 SM, 2 1/2  SM)
        sm_match = re.search(r'\b([PM])?(\d+(?:\s+\d+/\d+)?|\d+/\d+)\s*SM\b', raw_metar)
        if sm_match:
            prefix_code = sm_match.group(1)
            value = sm_match.group(2)
            
            # Clean up double/multiple spaces in value
            value = re.sub(r'\s+', ' ', value)
            
            prefix = ""
            if prefix_code == 'P':
                prefix = "greater than "
            elif prefix_code == 'M':
                prefix = "less than "
                
            unit = "statute mile" if value in ["1", "1/2", "1/4", "3/4", "1/8", "5/8"] else "statute miles"
            visibility = f"{prefix}{value} {unit}"
        else:
            # 2. KM visibility (e.g., 10KM, 5KM)
            km_match = re.search(r'\b(\d+)\s*KM\b', raw_metar)
            if km_match:
                value = km_match.group(1)
                unit = "kilometer" if value == "1" else "kilometers"
                visibility = f"{value} {unit}"
            else:
                # 3. 4-digit meter visibility (e.g., 9999, 4000, 0800, 1500)
                # Often followed by directional suffix (e.g. 4000NE) or NDV
                meter_match = re.search(r'\b(\d{4})(?:NDV|[NSEW]{1,2})?\b', raw_metar)
                if meter_match:
                    value = meter_match.group(1)
                    if value == "9999":
                        visibility = "10 kilometers or greater"
                    else:
                        visibility = f"{int(value)} meters"
                else:
                    visibility = "Visibility data unclear"
    
    # Append weather conditions to visibility if present (e.g. Haze, Mist, Fog)
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
    for word in raw_metar.split()[2:]:  # skip station and time
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
    
    # 1. Parse Wind Numeric Details for Recommendation
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

    # 2. Parse Visibility in Meters for Flight Category and Recommendation
    visibility_m = 10000.0  # Default to 10km (VFR)
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

    # 3. Parse Cloud Ceiling in Feet for Flight Category and Recommendation
    ceiling_ft = 99999.0  # Default to infinite ceiling (no ceiling)
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

    # 4. Check for Significant Weather
    has_sig_wx = False
    sig_wx_codes = ['TS', 'RA', 'DZ', 'SN', 'FG', 'BR', 'HZ', 'FU', 'SQ', 'FC', 'GR', 'GS', 'PL']
    metar_words = raw_metar.split()
    for word in metar_words[2:]:  # skip station and type
        clean_word = re.sub(r'^[+-]|VC', '', word)
        for code in sig_wx_codes:
            if clean_word == code or (len(clean_word) > 2 and code in clean_word and not clean_word.endswith('KT') and not clean_word.endswith('SM') and not clean_word.startswith('Q') and not clean_word.startswith('A')):
                has_sig_wx = True
                break

    # 5. Determine Flight Category
    if visibility_m < 1600.0 or ceiling_ft < 500.0:
        flight_category = "LIFR"
    elif visibility_m < 5000.0 or ceiling_ft < 1000.0:
        flight_category = "IFR"
    elif visibility_m >= 8000.0 and ceiling_ft >= 3000.0 and not has_sig_wx:
        flight_category = "VFR"
    else:
        flight_category = "MVFR"

    # 6. Evaluate Student Pilot Personal Minimums
    reasons = []
    
    # Visibility (Min 5 SM / 8000m)
    if visibility_m < 8000.0:
        reasons.append(f"Visibility is below student pilot minimum of 5 SM (measured: {visibility_m/1609.34:.1f} SM / {visibility_m:.0f} meters)")
        
    # Ceiling (Min 3,000 ft)
    if ceiling_ft < 3000.0:
        reasons.append(f"Ceiling is below student pilot minimum of 3,000 feet (measured: {ceiling_ft:,.0f} feet)")
        
    # Wind limit (Max crosswind conservative limit: sustained wind speed > 10 KT)
    if wind_speed > 10:
        reasons.append(f"Sustained wind speed is {wind_speed} knots, which exceeds the student maximum crosswind limit of 10 knots (potential crosswind exceedance depending on runway alignment)")
        
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
        "flight_recommendation": flight_recommendation,
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
        # Pre-clean string of hidden non-breaking and zero-width spaces
        clean_response = _clean_json_string(ai_response)
        # Parse JSON
        data = json.loads(clean_response)
        
        # Standardize keys in case the model used variations (e.g. when response_schema is disabled)
        key_mappings = {
            "station_id": "station",
            "icao": "station",
            "time": "time_of_observation",
            "time_issued": "time_of_observation",
            "observation_time": "time_of_observation",
            "wind": "surface_wind",
            "wind_conditions": "surface_wind",
            "temperature": "temperature_dew_point",
            "dew_point": "temperature_dew_point",
            "altimeter": "qnh",
            "altimeter_setting": "qnh"
        }
        
        # Clean up and standardize keys & values
        cleaned_data = {}
        for k, v in data.items():
            mapped_k = key_mappings.get(k, k)
            if isinstance(v, dict):
                # Flatten dictionary into a spoken sentence
                cleaned_data[mapped_k] = ", ".join(f"{str(nk).replace('_', ' ')}: {nv}" for nk, nv in v.items())
            elif isinstance(v, list):
                # Flatten list
                items_str = []
                for item in v:
                    if isinstance(item, dict):
                        items_str.append(", ".join(f"{str(nk).replace('_', ' ')}: {nv}" for nk, nv in item.items()))
                    else:
                        items_str.append(str(item))
                cleaned_data[mapped_k] = "; ".join(items_str)
            else:
                cleaned_data[mapped_k] = str(v)
                
        # Handle special split temperature/dewpoint cases
        if "temperature" in data and "dew_point" in data:
            cleaned_data["temperature_dew_point"] = f"Temperature {data['temperature']} degrees Celsius, dew point {data['dew_point']} degrees Celsius"
        
        # Validate against Pydantic schema
        validated = METARAnalysis(**cleaned_data)
        
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
            partial_data = json.loads(clean_response)
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
            "name": "openai",
            "key": OPENAI_KEY,
            "callable": _call_openai,
            "emoji": "🧠"
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
                    
                    if response and response.get("status") in ["success", "partial"]:
                        response["rate_limit_stats"] = rate_limiter.get_stats()
                        
                        # If it's a fallback provider, mark fallback_used
                        if provider["name"] != "gemini":
                            response["fallback_used"] = True
                            
                        return response
                    else:
                        print(f"   ⚠️ {provider['name']} validation failed, trying next provider...")
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
        "openai_configured": bool(OPENAI_KEY),
        "deepseek_available": bool(DEEPSEEK_KEY and rate_limiter.can_call("deepseek")),
        "gemini_available": bool(GEMINI_KEY and rate_limiter.can_call("gemini")),
        "groq_available": bool(GROQ_KEY and rate_limiter.can_call("groq")),
        "openai_available": bool(OPENAI_KEY and rate_limiter.can_call("openai")),
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