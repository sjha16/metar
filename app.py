import streamlit as st
import json
from io import BytesIO  
from gtts import gTTS   
from src.fetcher import fetch_metar, fetch_taf
from src.agent import analyze_metar_orchestrator

# Configure page layout for quick mobile responsiveness
st.set_page_config(page_title="ATC METAR & TAF Briefing", page_icon="🛫", layout="centered")

st.title("🛫 ATC Weather Assistant")
st.caption("AI-powered structural pilot briefing sequencer.")

# 1. INITIALIZE PERSISTENT APP MEMORY
# Streamlit clears variables on rerun, so we store the briefing data in session_state 
# so it doesn't disappear when we generate the voice later.
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "station_loaded" not in st.session_state:
    st.session_state.station_loaded = ""

icao = st.text_input("Enter Airport ICAO Code:", value="VECC").upper().strip()

# Handle the main weather fetch block
if st.button("Generate Pilot Briefing", type="primary"):
    if len(icao) != 4:
        st.error("Please enter a valid 4-letter ICAO code.")
    else:
        with st.spinner("Fetching raw logs and generating textual briefing..."):
            raw_metar = fetch_metar(icao)
            raw_taf = fetch_taf(icao)
            
            if "Error" in raw_metar or "Error" in raw_taf:
                st.error(f"Data Fetch Failure. METAR: {raw_metar} | TAF: {raw_taf}")
            else:
                # Run the dual-engine orchestrator
                analysis_json_str = analyze_metar_orchestrator(raw_metar, raw_taf)
                
                # Cache results in application memory
                st.session_state.analysis = json.loads(analysis_json_str)
                st.session_state.station_loaded = icao
                st.session_state.raw_metar = raw_metar
                st.session_state.raw_taf = raw_taf

# 2. RENDER THE CONTENT FROM MEMORY (FAST AND CRISP)
if st.session_state.analysis and st.session_state.station_loaded == icao:
    analysis = st.session_state.analysis
    
    if "error" in analysis:
        st.error(analysis["error"])
    else:
        st.success(f"Analysis Complete for {icao}!")
        
        # Display original raw texts right at the top
        st.subheader("📝 Original Raw Operational Logs")
        st.text_area("Live METAR:", value=st.session_state.raw_metar, height=70, disabled=True)
        st.text_area("Current TAF Block:", value=st.session_state.raw_taf, height=120, disabled=True)
        
        st.divider()
        
        # AI Deciphered Section Header
        st.subheader(f"📊 {analysis.get('station')} Deciphered Briefing Sequence")
        
        # Flight Category Banner Check
        category = analysis.get("flight_category", "UNKNOWN")
        if category in ["IFR", "LIFR"]:
            st.error(f"🔴 CURRENT FLIGHT CATEGORY: {category} (Instrument Rules Active)")
        elif category == "MVFR":
            st.warning(f"🟡 CURRENT FLIGHT CATEGORY: {category}")
        else:
            st.success(f"🟢 CURRENT FLIGHT CATEGORY: {category} (Visual Rules)")
        
        # --- OPTIMIZED AUDIO LAZY-LOADING BUTTON ---
        # Instead of auto-generating, we let the controller choose when to spend time making the MP3
        if st.button("🔊 Read Aloud (Generate Voice Summary)"):
            with st.spinner("Synthesizing clear audio speech stream..."):
                speech_text = (
                    f"Operational weather briefing for station {analysis.get('station')}. "
                    f"Current flight category is {category}. "
                    f"Observation time {analysis.get('time_of_observation')}. "
                    f"Surface wind {analysis.get('surface_wind')}. "
                    f"Visibility is {analysis.get('visibility')}. "
                    f"Cloud layers, {analysis.get('clouds')}. "
                    f"Temperature and dew point, {analysis.get('temperature_dew_point')}. "
                    f"Altimeter setting, {analysis.get('qnh')}. "
                    f"Upcoming trends. {analysis.get('forecast_trend')}. "
                    f"Pertinent operational information. {analysis.get('pertinent_information')}."
                )
                try:
                    tts = gTTS(text=speech_text, lang='en', slow=False)
                    fp = BytesIO()
                    tts.write_to_fp(fp)
                    fp.seek(0)
                    st.audio(fp, format="audio/mp3", autoplay=True) # Autoplays once generated successfully
                except Exception as audio_err:
                    st.error(f"Voice generation failed: {audio_err}")
        
        st.divider()
        
        # Render the text briefing cards layout instantly
        st.markdown(f"**⏰ Observation Time:** {analysis.get('time_of_observation')}")
        st.markdown(f"**💨 Surface Wind:** {analysis.get('surface_wind')}")
        st.markdown(f"**👁️ Visibility:** {analysis.get('visibility')}")
        st.markdown(f"**☁️ Clouds:** {analysis.get('clouds')}")
        st.markdown(f"**🌡️ Temp / Dew Point:** {analysis.get('temperature_dew_point')}")
        st.markdown(f"**🛑 QNH / Altimeter:** {analysis.get('qnh')}")
        
        st.divider()
        st.markdown(f"**🔮 Upcoming Forecast Trends (TAF Summary):**\n\n{analysis.get('forecast_trend')}")
        
        st.divider()
        st.markdown(f"**⚠️ Pertinent Operational Notes:**\n\n_{analysis.get('pertinent_information')}_")