import requests

def fetch_metar(icao: str) -> str:
    """Fetches the latest raw METAR string from the Aviation Weather Center API."""
    url = f"https://aviationweather.gov/api/data/metar?ids={icao.upper()}&format=raw"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200 and response.text.strip():
            return response.text.strip()
        return f"Error: Could not retrieve data for {icao}."
    except Exception as e:
        return f"Exception: {str(e)}"
    
def fetch_taf(icao: str) -> str:
    """Fetches the latest raw TAF text block for a given ICAO code."""
    url = f"https://aviationweather.gov/api/data/taf?ids={icao}&format=raw"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200 and response.text.strip():
            return response.text.strip()
        return f"Error: Could not find TAF data for {icao}."
    except Exception as e:
        return f"Exception occurred fetching TAF: {str(e)}"