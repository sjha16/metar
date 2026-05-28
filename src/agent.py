import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Import Gemini Tools
from google import genai
from google.genai import types
from google.genai.errors import APIError

# Import Groq (Using its native client package which supports OpenAI's structure)
from groq import Groq

load_dotenv()

class METARAnalysis(BaseModel):
    station: str = Field(description="The 4-letter ICAO code.")
    time_of_observation: str = Field(description="The time of observation extracted from the METAR (e.g., 0530 Zulu).")
    surface_wind: str = Field(description="Wind direction and speed, including any gusts or crosswind warnings.")
    visibility: str = Field(description="Visibility values, including explanations for drops (like haze or mist).")
    clouds: str = Field(description="Cloud layers, ceilings, or vertical visibility description.")
    temperature_dew_point: str = Field(description="The air temperature and dew point values.")
    qnh: str = Field(description="The altimeter setting / QNH value.")
    flight_category: str = Field(description="Strict operational category: VFR, MVFR, IFR, or LIFR.")
    pertinent_information: str = Field(description="Critical ATC takeaways (e.g., wind shear, ILS required, runway changes, or NOSIG context).")

# SYSTEM_INSTRUCTION = (
#     "You are an expert Air Traffic Control assistant. "
#     "Analyze the METAR text accurately. Fill out every field in the schema sequentially. "
#     "Ensure your summaries use standard aviation terminology used during pilot briefings."
# )
# --- IMPROVED SYSTEM INSTRUCTIONS WITH FEW-SHOT EXAMPLES ---
SYSTEM_INSTRUCTION = (
    "You are an expert Air Traffic Control assistant. Your job is to decode raw METAR data "
    "into beautiful, fully written-out aviation phraseology exactly as used during pilot briefings.\n\n"
    
    "CRITICAL RULE FOR FLIGHT CATEGORY:\n"
    "- If visibility is LESS than 5000 meters OR ceiling is LESS than 1000 feet, you MUST classify it as IFR.\n"
    "- Do not just read back raw text segments. You must translate raw code to spoken English text.\n\n"
    
    "EXAMPLE TRALSLATION:\n"
    "Raw Input: 'METAR VECC 280530Z 11007KT 3800 HZ SCT020 SCT100 32/26 Q1007 NOSIG'\n"
    "Target JSON Output Layout:\n"
    "{\n"
    "  \"station\": \"VECC\",\n"
    "  \"time_of_observation\": \"280530 Zulu\",\n"
    "  \"surface_wind\": \"Wind from one one zero degrees at seven knots.\",\n"
    "  \"visibility\": \"Three thousand eight hundred meters, reduced by haze.\",\n"
    "  \"clouds\": \"Scattered clouds at two thousand feet and scattered clouds at ten thousand feet.\",\n"
    "  \"temperature_dew_point\": \"Temperature thirty-two degrees Celsius, dew point twenty-six degrees Celsius.\",\n"
    "  \"qnh\": \"Altimeter setting one zero zero seven hectopascals.\",\n"
    "  \"flight_category\": \"IFR\",\n"
    "  \"pertinent_information\": \"No significant changes expected (NOSIG). Visibility below 5km triggers instrument procedures.\"\n"
    "}"
)

def analyze_metar_with_gemini(raw_metar: str) -> str:
    """Primary attempt using Gemini 2.5 Flash."""
    print("🤖 Attempting analysis with Gemini...")
    client = genai.Client()
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=f"Analyze this raw METAR string and organize it into a structured sequence: {raw_metar}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=METARAnalysis,
            temperature=0.0
        ),
    )
    return response.text
def analyze_metar_with_gemini(raw_metar: str) -> str:
    """Primary attempt using Gemini 2.5 Flash (Free Tier)."""
    print("🤖 Attempting analysis with Gemini...")
    client = genai.Client()
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=f"Analyze this raw METAR string and organize it into a structured sequence: {raw_metar}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=METARAnalysis,
            temperature=0.0
        ),
    )
    return response.text

# def analyze_metar_with_groq(raw_metar: str) -> str:
#     """Backup attempt using Free Groq Llama Cloud."""
#     print("🔄 Gemini busy. Switching to Groq (Llama 3) Free Backup...")
    
#     # Automatically picks up GROQ_API_KEY from your .env
#     client = Groq() 
    
#     # We use chat completions with structural output constraints
#     response = client.chat.completions.create(
#         model="llama-3.3-70b-versatile",
#         messages=[
#             {"role": "system", "content": SYSTEM_INSTRUCTION},
#             {"role": "user", "content": f"Analyze this raw METAR string and organize it into a structured sequence: {raw_metar}"}
#         ],
#         # Enforces the identical Pydantic output structure
#         response_format={"type": "json_object", "schema": METARAnalysis.model_json_schema()},
#         temperature=0.0
#     )
#     return response.choices[0].message.content
def analyze_metar_with_groq(raw_metar: str) -> str:
    """Backup attempt using Free Groq Llama Cloud with standard JSON mode."""
    print("Core: Switching to Groq (Llama 3) for live verification...")
    client = Groq() 
    
    # We add a hint in the prompt to ensure it sticks strictly to the keys we want
    schema_fields = list(METARAnalysis.model_fields.keys())
    prompt_with_schema = (
        f"Analyze this raw METAR string: '{raw_metar}'. "
        f"You MUST return a JSON object containing exactly these keys in order: {schema_fields}."
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt_with_schema}
        ],
        # Change type from "json_schema" to "json_object"
        response_format={"type": "json_object"},
        temperature=0.0
    )
    return response.choices[0].message.content

def analyze_metar_orchestrator(raw_metar: str) -> str:
    """Tries Gemini first, switches to Groq automatically if Gemini is overloaded."""
    try:
        return analyze_metar_with_gemini(raw_metar)
    except APIError as e:
        print(f"⚠️ Gemini API unavailable (Status {e.code}). Initiating free failover protocol...")
        try:
            return analyze_metar_with_groq(raw_metar)
        except Exception as groq_err:
            print(f"❌ Both free AI providers failed. Groq Error: {groq_err}")
            return '{"error": "All free AI models are currently rate-limited or busy."}'
    except Exception as general_err:
        print(f"⚠️ Unexpected system error: {general_err}")
        return '{"error": "Processing failed."}'

# def analyze_metar_orchestrator(raw_metar: str) -> str:
#     """TEMPORARY FORCED TESTING OF GROQ BACKUP"""
#     try:
#         # Comment this line out to bypass Gemini entirely for this test run:
#         # return analyze_metar_with_gemini(raw_metar)
        
#         # Directly call Groq to see if the API key and structure are working:
#         return analyze_metar_with_groq(raw_metar)
        
#     except Exception as groq_err:
#         print(f"❌ Groq Error: {groq_err}")
#         return '{"error": "All free AI models are currently rate-limited or busy."}'