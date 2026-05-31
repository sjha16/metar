import os
from pathlib import Path  # Standard library to calculate clean paths
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Import Gemini Tools
from google import genai
from google.genai import types
from google.genai.errors import APIError

# Import Groq
from groq import Groq

# 1. BULLETPROOF LOCAL API KEY LOADING
# This calculates your project root dynamically and points directly to the .env file
base_dir = Path(__file__).resolve().parent.parent
env_path = base_dir / ".env"
load_dotenv(dotenv_path=env_path)

# Extract variables from environment
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")

# =====================================================================
# THE REST OF YOUR CODE BELOW IS 100% CORRECT. KEEP IT EXACTLY AS IS:
# =====================================================================

class METARAnalysis(BaseModel):
    station: str = Field(description="The 4-letter ICAO code.")
    time_of_observation: str = Field(description="The time of observation extracted from the METAR (e.g., 0530 Zulu).")
    surface_wind: str = Field(description="Wind direction and speed, fully decoded into spoken text (e.g., 'one one zero degrees at six knots').")
    visibility: str = Field(description="Visibility values fully decoded into words, explaining any drops (e.g., 'three thousand eight hundred meters due to haze').")
    clouds: str = Field(description="Cloud layers and ceilings fully decoded into spoken text.")
    temperature_dew_point: str = Field(description="The air temperature and dew point values written out in full words.")
    qnh: str = Field(description="The altimeter setting / QNH value written out fully.")
    flight_category: str = Field(description="Strict current operational category: VFR, MVFR, IFR, or LIFR.")
    forecast_trend: str = Field(description="A clear summary of upcoming weather changes, wind shifts, or category upgrades/downgrades expected over the next few hours based on the TAF.")
    pertinent_information: str = Field(description="Critical ATC takeaways (e.g., wind shear, ILS required, runway changes, or TEMPO hazards).")

SYSTEM_INSTRUCTION = (
    "You are an expert Air Traffic Control assistant. Your job is to decode raw METAR and TAF data "
    "into beautiful, fully written-out aviation phraseology exactly as used during pilot briefings.\n\n"
    "CRITICAL RULE FOR FLIGHT CATEGORY:\n"
    "- Evaluate the CURRENT visibility and ceilings from the METAR.\n"
    "- If visibility is LESS than 5000 meters OR ceiling is LESS than 1000 feet, you MUST classify it as IFR.\n"
    "- Translate alphanumeric code elements completely into spoken English text formats.\n"
)

def analyze_metar_with_gemini(raw_metar: str, raw_taf: str) -> str:
    """Primary attempt using Gemini 2.5 Flash."""
    print("🤖 Attempting analysis with Gemini...")
    client = genai.Client(api_key=GEMINI_KEY)
    
    combined_content = f"Raw METAR string: {raw_metar}\nRaw TAF block: {raw_taf}"
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=f"Analyze this raw airport weather data and organize it into a structured sequence:\n{combined_content}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=METARAnalysis,
            temperature=0.0
        ),
    )
    return response.text

def analyze_metar_with_groq(raw_metar: str, raw_taf: str) -> str:
    """Backup attempt using Free Groq Llama Cloud."""
    print("Core: Switching to Groq (Llama 3) for live verification...")
    client = Groq(api_key=GROQ_KEY) 
    
    combined_content = f"Raw METAR string: {raw_metar}\nRaw TAF block: {raw_taf}"
    schema_fields = list(METARAnalysis.model_fields.keys())
    
    prompt_with_schema = (
        f"Analyze this raw airport weather data:\n{combined_content}\n\n"
        f"You MUST return a JSON object containing exactly these keys: {schema_fields}. "
        f"Follow the explicit verbal translation instructions and calculation rules given in your system instructions."
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt_with_schema}
        ],
        response_format={"type": "json_object"},
        temperature=0.0
    )
    return response.choices[0].message.content

def analyze_metar_orchestrator(raw_metar: str, raw_taf: str) -> str:
    try:
        return analyze_metar_with_gemini(raw_metar, raw_taf)
    except APIError as e:
        print(f"⚠️ Gemini API unavailable (Status {e.code}). Initiating free failover protocol...")
        try:
            return analyze_metar_with_groq(raw_metar, raw_taf)
        except Exception as groq_err:
            print(f"❌ Both free AI providers failed. Groq Error: {groq_err}")
            return '{"error": "All free AI models are currently rate-limited or busy."}'
    except Exception as general_err:
        print(f"⚠️ Unexpected system error: {general_err}")
        return '{"error": "Processing failed."}'