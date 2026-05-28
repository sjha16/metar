import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# Load the environment variables from the .env file
load_dotenv()

class METARAnalysis(BaseModel):
    station: str = Field(description="The 4-letter ICAO code.")
    flight_category: str = Field(description="VFR, MVFR, IFR, or LIFR.")
    hazards: list[str] = Field(description="Critical hazards like high crosswinds, low visibility, thunderstorms, etc.")
    atc_summary: str = Field(description="A sharp, operational takeaway for an ATC.")

def analyze_metar_with_gemini(raw_metar: str) -> str:
    # client automatically picks up GEMINI_API_KEY from the environment
    client = genai.Client()
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=f"Analyze this raw METAR string and extract operational metrics: {raw_metar}",
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are an expert Air Traffic Control automation assistant. "
                "Analyze the METAR string accurately. Evaluate cloud ceilings and visibility "
                "to determine flight categories strictly by aviation standards."
            ),
            response_mime_type="application/json",
            response_schema=METARAnalysis,
            temperature=0.0
        ),
    )
    return response.text