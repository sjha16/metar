import streamlit as st
import json
from src.fetcher import fetch_metar
from src.agent import analyze_metar_orchestrator

# 1. Configure the page layout for mobile compatibility
st.set_page_config(page_title="ATC METAR Briefing", page_icon="🛫", layout="centered")

st.title("🛫 ATC METAR Assistant")
st.caption("AI-powered structural pilot briefing sequencer using Gemini & Groq.")

# 2. Add an input box for the ICAO code
icao = st.text_input("Enter Airport ICAO Code:", value="VECC").upper().strip()

if st.button("Generate Pilot Briefing", type="primary"):
    if len(icao) != 4:
        st.error("Please enter a valid 4-letter ICAO code.")
    else:
        with st.spinner(f"Fetching raw data and processing with AI fallback engine..."):
            # Fetch raw data using your existing tool
            raw_data = fetch_metar(icao)
            
            if "Error" in raw_data or "Exception" in raw_data:
                st.error(raw_data)
            else:
                # Process data through your dual-engine failover system
                analysis_json_str = analyze_metar_orchestrator(raw_data)
                analysis = json.loads(analysis_json_str)
                
                if "error" in analysis:
                    st.error(analysis["error"])
                else:
                    st.success(f"Analysis Complete for {icao}!")
                    
                    # --- NEW TOP SECTION: RAW METAR DISPLAY ---
                    st.subheader("📝 Original METAR String")
                    st.code(raw_data, language="text")
                    
                    st.divider() # Clean visual separation line
                    
                    # --- BOTTOM SECTION: AI DECIPHERED SEQUENCE ---
                    st.subheader(f"📊 {analysis.get('station')} Deciphered Operational Briefing")
                    
                    # Highlight the Flight Category cleanly
                    category = analysis.get("flight_category", "UNKNOWN")
                    if category in ["IFR", "LIFR"]:
                        st.error(f"🔴 FLIGHT CATEGORY: {category} (Instrument Rules Active)")
                    elif category == "MVFR":
                        st.warning(f"🟡 FLIGHT CATEGORY: {category}")
                    else:
                        st.success(f"🟢 FLIGHT CATEGORY: {category} (Visual Rules)")
                        
                    # Display the parameters in your exact pilot-readback sequence
                    st.markdown(f"**⏰ Observation Time:** {analysis.get('time_of_observation')}")
                    st.markdown(f"**💨 Surface Wind:** {analysis.get('surface_wind')}")
                    st.markdown(f"**👁️ Visibility:** {analysis.get('visibility')}")
                    st.markdown(f"**☁️ Clouds:** {analysis.get('clouds')}")
                    st.markdown(f"**🌡️ Temp / Dew Point:** {analysis.get('temperature_dew_point')}")
                    st.markdown(f"**🛑 QNH / Altimeter:** {analysis.get('qnh')}")
                    
                    st.divider()
                    st.markdown(f"**⚠️ Pertinent Operational Notes:**\n\n_{analysis.get('pertinent_information')}_")