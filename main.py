import sys
from src.fetcher import fetch_metar
from src.agent import analyze_metar_orchestrator

def main():
    # Use VECC (Kolkata) as default, or accept an airport via command line
    icao = sys.argv[1] if len(sys.argv) > 1 else "VECC"
    
    print(f"📡 Fetching METAR for {icao.upper()}...")
    raw_data = fetch_metar(icao)
    
    if "Error" in raw_data or "Exception" in raw_data:
        print(raw_data)
        return
        
    print(f"📝 Raw Data: {raw_data}\n")
    
    print("🤖 Processing with AI...")
    analysis_result = analyze_metar_orchestrator(raw_data)
    
    print("\n📋 Final Structured Result:")
    
    # This automatically splits the long single line into a beautiful, readable format
    import json
    readable_json = json.dumps(json.loads(analysis_result), indent=4)
    print(readable_json)

if __name__ == "__main__":
    main()