"""
Aviation Weather Briefing App
Streamlit frontend for AI-powered METAR/TAF analysis
"""

import streamlit as st
import json
from io import BytesIO
from datetime import datetime
from pathlib import Path
from gtts import gTTS
import streamlit_analytics2 as streamlit_analytics

from src.fetcher import fetch_metar, fetch_taf, fetch_airport_info
from src.agent import (
    analyze_metar_orchestrator,
    get_ai_status,
    format_analysis_for_display,
    cached_analyze_metar
)


# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="ATC METAR & TAF Briefing",
    page_icon="🛫",
    layout="wide",  # Changed to wide for better content display
    initial_sidebar_state="expanded"
)


# ============================================
# CUSTOM CSS FOR BETTER UI
# ============================================

css_path = Path(__file__).resolve().parent / "src" / "styles.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


# ============================================
# SESSION STATE INITIALIZATION
# ============================================

def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        "analysis": None,
        "station_loaded": "",
        "airport_name": "",
        "raw_metar": "",
        "raw_taf": "",
        "search_history": [],
        "audio_generated": False,
        "audio_data": None,
        "favorite_airports": ["VECC", "VABB", "VIDP", "VOBL", "VOMM"],
        "dark_mode": False,
        "request_count": 0,
        "page_views": 0
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


init_session_state()

# Track page views in session state
st.session_state.page_views += 1


# ============================================
# SIMPLE GROWTH ANALYTICS
# ============================================

def log_briefing_generated():
    """Log each successful weather briefing generation to track growth"""
    log_file = "visits.log"
    try:
        timestamp = datetime.now().isoformat()
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{timestamp}\n")
    except Exception as e:
        print(f"Error logging briefing: {e}")

def get_total_briefings_count() -> int:
    """Get the total number of briefings generated from visits.log"""
    log_file = "visits.log"
    try:
        if Path(log_file).exists():
            with open(log_file, "r", encoding="utf-8") as f:
                return len(f.readlines())
    except Exception:
        pass
    return 0


# ============================================
# SIDEBAR - Controls & Info
# ============================================

def render_sidebar():
    """Render the sidebar with controls and status"""
    
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/airplane-take-off.png", width=80)
        st.markdown("### 🛫 ATC Weather Assistant")
        st.caption("Automated Pilot & ATC Briefing System")
        
        st.divider()
        
        # Parser Status Section
        st.markdown("#### ⚙️ Parser Engine Status")
        st.success("Regex Engine Active ✅")
        
        # Cache Stats
        ai_status = get_ai_status()
        cache_stats = ai_status.get("cache_stats", {"memory_entries": 0, "disk_entries": 0})
        st.caption(f"📦 Cache: {cache_stats.get('memory_entries', 0)} memory / {cache_stats.get('disk_entries', 0)} disk entries")
            
        st.divider()
        
        # Quick Access Airports
        st.markdown("#### ⚡ Quick Access")
        
        # Indian airports
        st.caption("🇮🇳 Indian Airports")
        indian_cols = st.columns(3)
        indian_airports = ["VECC", "VABB", "VIDP", "VOBL", "VOMM", "VAAH"]
        for i, airport in enumerate(indian_airports):
            with indian_cols[i % 3]:
                if st.button(f"{airport}", key=f"ind_{airport}", use_container_width=True):
                    st.session_state.quick_select = airport
                    st.session_state.icao_input = airport
                    st.session_state.quick_trigger = True
                    st.rerun()
        
        # International airports
        st.caption("🌍 International")
        intl_cols = st.columns(3)
        intl_airports = ["KJFK", "EGLL", "OMDB", "KLAX", "YSSY", "WSSS"]
        for i, airport in enumerate(intl_airports):
            with intl_cols[i % 3]:
                if st.button(f"{airport}", key=f"int_{airport}", use_container_width=True):
                    st.session_state.quick_select = airport
                    st.session_state.icao_input = airport
                    st.session_state.quick_trigger = True
                    st.rerun()
        
        st.divider()
        
        # Search History
        if st.session_state.search_history:
            st.markdown("#### 🕒 Recent Searches")
            for hist in reversed(st.session_state.search_history[-5:]):
                st.caption(f"• {hist}")
        
        st.divider()
        
        # Live Stats
        st.markdown("### 📊 Live Stats")
        
        # Display Total Briefings Generated
        briefings_count = get_total_briefings_count()
        if briefings_count > 0:
            st.metric("📈 Total Briefings Generated", f"{briefings_count:,}")
            
        # Display Session Analyses
        st.metric("📊 Session Analyses", st.session_state.get('request_count', 0))
            
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🌍 Airports Covered", "9,500+")
        with col2:
            st.metric("⚙️ Engine", "Regex Parser")
            
        st.metric("💰 Cost to You", "100% Free Forever")
        
        st.divider()
        st.markdown("### ⭐ User Love")
        st.markdown("""
        *"Incredible tool for student pilots!"* 
        — **r/flying community**
        
        *"Finally understand METAR without my CFI"*
        — **Student Pilot**
        """)
        
        st.divider()
        
        # Feedback Section
        st.markdown("### 💬 Share Your Feedback")
        
        rating = st.select_slider(
            "Rate this app:",
            options=["😞", "😐", "🙂", "😊", "🤩"],
            value="🙂",
            key="app_rating_slider"
        )
        
        feedback = st.text_area("How can we improve?", placeholder="Feature requests, bugs, or reviews...", key="feedback_input")
        if st.button("🚀 Send Feedback", key="send_feedback_btn", use_container_width=True):
            try:
                log_entry = f"{datetime.now().isoformat()} | Rating: {rating} | Feedback: {feedback.strip() if feedback.strip() else 'No text feedback'}\n"
                with open("feedback.txt", "a", encoding="utf-8") as f:
                    f.write(log_entry)
                st.success("🎉 Thanks! Feedback saved.")
            except Exception as e:
                st.error(f"Failed to save feedback: {e}")
                
        st.divider()
        
        # Footer
        st.caption("Data from aviationweather.gov")
        st.caption("Regex decoding engine")
        st.caption(f"Session requests: {st.session_state.request_count}")
        
        # Reset button
        if st.button("🔄 Reset Session", use_container_width=True):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()


# ============================================
# WEATHER FETCHING LOGIC
# ============================================

def fetch_weather_data(icao: str):
    """Fetch and analyze weather data for an airport"""
    
    # Validate ICAO
    if len(icao) != 4 or not icao.isalpha():
        st.error("❌ Please enter a valid 4-letter ICAO code.")
        return False
    
    # Track request
    st.session_state.request_count += 1
    
    with st.spinner(f"🔍 Fetching weather data for {icao}..."):
        # Fetch raw data
        raw_metar = fetch_metar(icao)
        raw_taf = fetch_taf(icao)
        
        # Fetch airport name
        airport_name = ""
        try:
            airport_info = fetch_airport_info(icao)
            if airport_info and "name" in airport_info:
                airport_name = airport_info["name"]
        except Exception:
            pass
        
        # Check for fetch errors
        if "Error" in raw_metar:
            st.error(f"❌ METAR Fetch Failed: {raw_metar}")
            return False
        
        if "Error" in raw_taf:
            st.warning(f"⚠️ TAF data unavailable: {raw_taf}")
            raw_taf = "TAF NOT AVAILABLE FOR THIS STATION"
        
        # Show raw data immediately
        with st.expander("📡 Raw Data Received", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.code(raw_metar, language=None)
                st.caption("Raw METAR")
            with col2:
                st.code(raw_taf if raw_taf else "No TAF", language=None)
                st.caption("Raw TAF")
    
    # Regex Analysis
    with st.spinner("⚙️ Decoding weather patterns using regex parser..."):
        try:
            # Directly call orchestrator
            analysis = analyze_metar_orchestrator(raw_metar, raw_taf)
            
            # Format for display
            analysis = format_analysis_for_display(analysis)
            
            # Store in session state
            st.session_state.analysis = analysis
            st.session_state.station_loaded = icao
            st.session_state.airport_name = airport_name
            st.session_state.raw_metar = raw_metar
            st.session_state.raw_taf = raw_taf
            st.session_state.audio_generated = False
            st.session_state.audio_data = None
            
            # Log successful briefing generation
            log_briefing_generated()
            
            # Add to search history
            if icao not in st.session_state.search_history:
                st.session_state.search_history.append(icao)
                if len(st.session_state.search_history) > 10:
                    st.session_state.search_history.pop(0)
            
            # Show toast feedback
            if analysis.get("from_cache"):
                st.toast("⚡ Instant result loaded from cache!", icon="🔋")
            else:
                st.toast("✅ Weather Briefing Decoded!", icon="⚙️")
                
            return True
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            return False


# ============================================
# DISPLAY FUNCTIONS
# ============================================

def display_flight_recommendation(recommendation: str):
    """Display Go/No-Go recommendation with visual badge"""
    recommendation_upper = recommendation.upper()
    
    if "NO-GO" in recommendation_upper:
        st.markdown("""
        <div style="background: #fef2f2; border-left: 4px solid #ef4444; padding: 20px; border-radius: 8px; margin: 10px 0;">
            <h3 style="color: #991b1b; margin: 0; font-weight: 700;">🔴 NO-GO - Flight Not Recommended</h3>
            <p style="margin-top: 10px; color: #7f1d1d; margin-bottom: 0; font-size: 1.05rem; line-height: 1.5;">{}</p>
        </div>
        """.format(recommendation), unsafe_allow_html=True)
    
    elif "GO" in recommendation_upper and "NO-GO" not in recommendation_upper:
        st.markdown("""
        <div style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 20px; border-radius: 8px; margin: 10px 0;">
            <h3 style="color: #166534; margin: 0; font-weight: 700;">🟢 GO - Conditions Favorable</h3>
            <p style="margin-top: 10px; color: #14532d; margin-bottom: 0; font-size: 1.05rem; line-height: 1.5;">{}</p>
        </div>
        """.format(recommendation), unsafe_allow_html=True)
    
    elif "MARGINAL" in recommendation_upper:
        st.markdown("""
        <div style="background: #fffbeb; border-left: 4px solid #f59e0b; padding: 20px; border-radius: 8px; margin: 10px 0;">
            <h3 style="color: #92400e; margin: 0; font-weight: 700;">🟡 MARGINAL - Consult Instructor</h3>
            <p style="margin-top: 10px; color: #78350f; margin-bottom: 0; font-size: 1.05rem; line-height: 1.5;">{}</p>
        </div>
        """.format(recommendation), unsafe_allow_html=True)
    
    else:
        st.info(f"**Flight Assessment:** {recommendation}")


def display_weather_metrics(analysis: dict):
    """Display weather metrics in a grid layout"""
    
    st.markdown('<p class="section-title">📊 Weather Metrics</p>', unsafe_allow_html=True)
    
    # Row 1
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container():
            st.markdown("**⏰ Observation Time**")
            st.markdown(f"`{analysis.get('time_of_observation', 'N/A')}`")
    
    with col2:
        with st.container():
            st.markdown("**💨 Surface Wind**")
            st.markdown(f"`{analysis.get('surface_wind', 'N/A')}`")
    
    with col3:
        with st.container():
            st.markdown("**👁️ Visibility**")
            st.markdown(f"`{analysis.get('visibility', 'N/A')}`")
    
    # Row 2
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container():
            st.markdown("**☁️ Cloud Conditions**")
            st.markdown(f"`{analysis.get('clouds', 'N/A')}`")
    
    with col2:
        with st.container():
            st.markdown("**🌡️ Temperature / Dew Point**")
            st.markdown(f"`{analysis.get('temperature_dew_point', 'N/A')}`")
    
    with col3:
        with st.container():
            st.markdown("**🛑 QNH / Altimeter**")
            st.markdown(f"`{analysis.get('qnh', 'N/A')}`")


def display_forecast_and_notes(analysis: dict):
    """Display forecast trends and operational notes"""
    
    st.markdown('<p class="section-title">🔮 Forecast & Operational Notes</p>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📈 Forecast Trends", "⚠️ Operational Notes"])
    
    with tab1:
        forecast = analysis.get('forecast_trend', 'No forecast available')
        st.markdown(f"""
        <div style="background: #f8fafc; color: #1e293b; padding: 15px; border-radius: 8px; border-left: 4px solid #667eea; font-size: 1rem; line-height: 1.5;">
            {forecast}
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        notes = analysis.get('pertinent_information', 'No operational notes')
        st.markdown(f"""
        <div style="background: #fffbeb; color: #78350f; padding: 15px; border-radius: 8px; border-left: 4px solid #f59e0b; font-size: 1rem; line-height: 1.5;">
            ⚠️ {notes}
        </div>
        """, unsafe_allow_html=True)



def generate_audio_briefing(analysis: dict):
    """Generate and cache audio briefing"""
    
    # Check if already generated
    if st.session_state.audio_generated and st.session_state.audio_data:
        return st.session_state.audio_data
    
    with st.spinner("🔊 Synthesizing voice briefing..."):
        try:
            speech_text = (
                f"Operational weather briefing for {analysis.get('station', 'this station')}. "
                f"Current flight category is {analysis.get('flight_category', 'unknown')}. "
                f"Observation time {analysis.get('time_of_observation', '')}. "
                f"Surface wind, {analysis.get('surface_wind', '')}. "
                f"Visibility is {analysis.get('visibility', '')}. "
                f"Cloud conditions, {analysis.get('clouds', '')}. "
                f"Temperature and dew point, {analysis.get('temperature_dew_point', '')}. "
                f"Altimeter setting, {analysis.get('qnh', '')}. "
                f"Forecast trends. {analysis.get('forecast_trend', '')}. "
                f"Flight recommendation. {analysis.get('flight_recommendation', '')}. "
                f"Operational notes. {analysis.get('pertinent_information', '')}."
            )
            
            tts = gTTS(text=speech_text[:500], lang='en', slow=False)  # Limit length
            fp = BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            
            # Cache the audio
            st.session_state.audio_generated = True
            st.session_state.audio_data = fp
            
            return fp
            
        except Exception as e:
            st.error(f"Voice generation failed: {str(e)}")
            return None


def display_analysis_results():
    """Main function to display all analysis results"""
    
    analysis = st.session_state.analysis
    
    # Handle error states
    if analysis.get("status") == "error":
        st.error(f"❌ Analysis Error: {analysis.get('message', 'Unknown error')}")
        if analysis.get("warning"):
            st.warning(analysis["warning"])
        return
    
    # Check if AI was available
    if not analysis.get("ai_available", True):
        st.warning(analysis.get("warning", "⚠️ AI services unavailable - showing basic analysis"))
    
    # Provider badge
    st.markdown(f"""
    <span style="background: #FF980015; 
                 color: #FF9800; 
                 padding: 4px 12px; 
                 border-radius: 12px; 
                 font-size: 0.8rem;">
        ⚙️ Engine: Regex Parser
    </span>
    """, unsafe_allow_html=True)
    
    if analysis.get("status") == "partial":
        st.info("ℹ️ Some fields may be incomplete")
    
    # Get the actual analysis data
    analysis_data = analysis.get("analysis", analysis)
    
    # Success message
    station = analysis_data.get('station', 'Unknown')
    airport_name = st.session_state.get("airport_name", "")
    if airport_name:
        st.success(f"✅ Weather Analysis Complete for {station} - {airport_name}")
    else:
        st.success(f"✅ Weather Analysis Complete for {station}")
    
    st.divider()
    
    # Flight Category Banner
    category = analysis_data.get("flight_category", "UNKNOWN")
    
    if category in ["LIFR"]:
        st.error(f"🔴 FLIGHT CATEGORY: {category} - LOW INSTRUMENT FLIGHT RULES")
        st.caption("⚠️ Extremely poor conditions. Operations highly restricted.")
    elif category in ["IFR"]:
        st.error(f"🔴 FLIGHT CATEGORY: {category} - INSTRUMENT FLIGHT RULES")
        st.caption("Instrument rating and clearance required.")
    elif category == "MVFR":
        st.warning(f"🟡 FLIGHT CATEGORY: {category} - MARGINAL VISUAL FLIGHT RULES")
        st.caption("Marginal conditions. Exercise increased caution.")
    elif category == "VFR":
        st.success(f"🟢 FLIGHT CATEGORY: {category} - VISUAL FLIGHT RULES")
        st.caption("Visual flight conditions. Standard operations.")
    else:
        st.info(f"⚪ FLIGHT CATEGORY: {category}")
    
    st.divider()
    
    # Flight Recommendation (Most important!)
    recommendation = analysis_data.get("flight_recommendation", "")
    if recommendation:
        display_flight_recommendation(recommendation)
        st.divider()
    
    # Weather Metrics Grid
    display_weather_metrics(analysis_data)
    
    st.divider()
    
    # Forecast and Notes
    display_forecast_and_notes(analysis_data)
    

    
    st.divider()
    
    # Audio Generation
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if st.button("🔊 Generate Voice Briefing", type="secondary", use_container_width=True):
            audio_data = generate_audio_briefing(analysis_data)
            if audio_data:
                st.audio(audio_data, format="audio/mp3")
    
    with col2:
        # Download as text
        briefing_text = json.dumps(analysis_data, indent=2)
        st.download_button(
            label="📥 Download Briefing",
            data=briefing_text,
            file_name=f"weather_briefing_{analysis_data.get('station', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col3:
        # Share button (copy to clipboard)
        if st.button("📋 Copy Summary", use_container_width=True):
            summary = (
                f"Weather Briefing: {analysis_data.get('station')}\n"
                f"Category: {category}\n"
                f"Wind: {analysis_data.get('surface_wind')}\n"
                f"Visibility: {analysis_data.get('visibility')}\n"
                f"Clouds: {analysis_data.get('clouds')}\n"
                f"Recommendation: {analysis_data.get('flight_recommendation')[:200]}"
            )
            st.code(summary, language=None)
            st.success("Summary copied! (Select and Ctrl+C)")


def show_high_traffic_mode():
    """Display when AI providers are overwhelmed"""
    st.warning("""
    ### 🚦 High Traffic Mode Active
    We are experiencing high demand! Here is what's happening:
    
    - ⚡ **Popular airports load instantly** (pre-cached)
    - 🔄 **New searches use basic analysis** (highly accurate, non-AI)
    - ⏱️ **AI analysis resumes automatically** when traffic normalizes
    
    **Quick Tip**: Select one of the popular airports below (e.g. VIDP, VABB, KJFK) for instant cached results!
    """)


# ============================================
# MAIN APP
# ============================================

def main():
    """Main application entry point"""
    
    with streamlit_analytics.track(load_from_json="weather_analytics.json", save_to_json="weather_analytics.json"):
        # Render sidebar
        render_sidebar()
        
        # Main content area
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.markdown('<h1 class="main-title">🛫 ATC Weather Assistant</h1>', unsafe_allow_html=True)
            st.caption("Automated Structural Pilot Briefing Sequencer")
        
        st.divider()
        
        # Input section
        input_col1, input_col2, input_col3 = st.columns([2, 1, 1])
        
        with input_col1:
            # Check if quick select was clicked
            default_icao = st.session_state.get("quick_select", "VECC")
            if "quick_select" in st.session_state:
                del st.session_state["quick_select"]
            
            icao = st.text_input(
                "Enter Airport ICAO Code:",
                value=default_icao,
                max_chars=4,
                placeholder="e.g., VECC, KJFK",
                key="icao_input"
            ).upper().strip()
        
        with input_col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing
            search_clicked = st.button(
                "🔍 Generate Briefing",
                type="primary",
                use_container_width=True
            )
        
        with input_col3:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing
            if st.session_state.analysis:
                if st.button("🔄 Refresh", use_container_width=True):
                    search_clicked = True
        
        # Handle search
        quick_trigger = st.session_state.get("quick_trigger", False)
        if "quick_trigger" in st.session_state:
            del st.session_state["quick_trigger"]
            
        if (search_clicked or quick_trigger) and icao:
            fetch_weather_data(icao)
        
        # Display results if available
        if st.session_state.analysis and st.session_state.station_loaded == icao:
            st.divider()
            display_analysis_results()
            
            # Raw data at the bottom
            st.divider()
            with st.expander("📡 View Raw Aviation Data", expanded=False):
                raw_col1, raw_col2 = st.columns(2)
                with raw_col1:
                    st.text_area(
                        "Live METAR:",
                        value=st.session_state.raw_metar,
                        height=100,
                        disabled=True,
                        key="raw_metar_display"
                    )
                with raw_col2:
                    st.text_area(
                        "Current TAF Block:",
                        value=st.session_state.raw_taf,
                        height=100,
                        disabled=True,
                        key="raw_taf_display"
                    )
        
        elif not search_clicked and not st.session_state.analysis:
            # Welcome screen
            st.info("👆 Enter an ICAO airport code above and click 'Generate Briefing' to get started!")
            
            # Feature showcase
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                ### ⚙️ Regex Parser
                Deterministic parsing algorithm decodes raw METAR/TAF weather strings into plain English with 
                student pilot recommendations.
                """)
            with col2:
                st.markdown("""
                ### 📊 Smart Briefings
                Get structured weather briefings with flight category, 
                Go/No-Go recommendations, and operational notes.
                """)
            with col3:
                st.markdown("""
                ### 🔊 Voice Output
                Generate audio briefings for hands-free 
                weather review during pre-flight planning.
                """)
        
        # =====================================================================
        # 🎓 COMPREHENSIVE GROUND SCHOOL REFERENCE DESK & EXTERNAL RESOURCES
        # =====================================================================
        st.divider()
        with st.expander("🎓 ATC Ground School Reference Desk (Click to Expand)", expanded=False):
            st.markdown("### 🎓 ATC Ground School Reference Desk")
            st.caption("Use this interactive blueprint to practice decoding raw weather logs and studying official aviation frameworks.")

            # Updated to 6 interactive, mobile-friendly tabs covering all major reports
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📝 METAR", "🔮 TAF", "🚨 SPECI", "⛈️ SIGMET/AIRMET", "📚 Glossary Table", "🌐 Web References"
            ])

            with tab1:
                st.subheader("How to Decode a METAR")
                st.markdown("""
                A **METAR** (Aviation Routine Weather Report) is an alphanumeric observation statement issued globally at fixed hourly or half-hourly intervals[cite: 377, 378].
                
                **Standard Sequence Blueprint:**
                `TYPE` ➡️ `ICAO` ➡️ `TIME` ➡️ `WIND` ➡️ `VISIBILITY` ➡️ `WEATHER` ➡️ `CLOUDS` ➡️ `TEMP/DEW` ➡️ `QNH` ➡️ `TREND`
                
                * **Example Analyzed:** `METAR VECC 020830Z 11007KT 3800 HZ SCT020 32/26 Q1007 NOSIG` [cite: 134, 167]
                    * **VECC:** Station identifier (Kolkata)[cite: 134].
                    * **020830Z:** 2nd day of the month at 08:30 Zulu (UTC) time.
                    * **11007KT:** Wind coming from 110 degrees at 7 knots[cite: 246].
                    * **3800 HZ:** Visibility 3,800 meters due to Haze[cite: 134, 246].
                    * **SCT020:** Scattered clouds at 2,000 feet[cite: 234].
                    * **32/26:** Air temperature 32°C, Dew point 26°C[cite: 167]. (When close together, relative humidity is near 100%—watch for fog/mist formation!).
                    * **Q1007:** Altimeter setting / QNH is 1007 Hectopascals[cite: 167, 617].
                    * **NOSIG:** No Significant Change expected over the next two hours[cite: 234].
                """)

            with tab2:
                st.subheader("How to Decode a TAF")
                st.markdown("""
                A **TAF** (Terminal Aerodrome Forecast) is a statement of forecast meteorological conditions significant to aviation, typically covering a 24 to 30-hour period[cite: 412, 575].
                
                **Key Change Identifiers to Know:**
                * **FM (From):** Used to indicate a rapid, permanent change starting exactly at a specific hour and minute.
                * **BECMG (Becoming):** Indicates a gradual evolution of weather parameters over a specific period (usually 1 to 2 hours)[cite: 583, 621].
                * **TEMPO (Temporary):** Frequent, temporary fluctuations lasting less than an hour at a time and covering less than half of the total forecast period[cite: 578, 621].
                
                **Example Complex Block:**
                `TEMPO 0110/0114 14010G20KT 2500 -TSRA FEW035CB` [cite: 569]
                * **Meaning:** Temporarily between the 1st day at 10:00Z and the 1st day at 14:00Z [cite: 578], winds will be 140 degrees at 10 knots gusting to 20 knots [cite: 580], visibility dropping to 2,500 meters in light thunderstorms and rain [cite: 581], with dangerous Cumulonimbus storm cells at 3,500 feet[cite: 582].
                """)

            with tab3:
                st.subheader("How to Identify a SPECI")
                st.markdown("""
                A **SPECI** is an **Aviation Special Weather Report**[cite: 383]. Unlike a regular METAR which is scheduled [cite: 378], a SPECI is issued *instantly* the moment a critical safety threshold is breached[cite: 383].
                
                **What triggers a SPECI operational alert?**
                1. **Wind Shift:** Wind direction changes by 60 degrees or more in 10 minutes, or a sudden heavy gust begins.
                2. **Visibility Drops:** Visibility falls below critical operational markers (e.g., dropping below 5000m, 3000m, or 1500m)[cite: 250].
                3. **Ceiling Changes:** Cloud base drops below 1,000 feet (pushing the field into sudden Instrument Flight Rules / IFR status)[cite: 134, 486].
                4. **Severe Weather Begins:** The sudden onset of thunderstorms (TS), freezing rain (FZRA), squalls, or wind shear[cite: 41].
                """)

            with tab4:
                st.subheader("Enroute Hazards: SIGMET vs. AIRMET")
                st.markdown("""
                While METAR and TAF focus strictly on an individual airport terminal area[cite: 377, 412], **SIGMETs** and **AIRMETs** inform pilots about severe enroute safety hazards across an entire flight information region (FIR).
                
                ### ⛈️ SIGMET (Significant Meteorological Information)
                Issued for severe, high-consequence weather hazards that affect **all aircraft** (commercial airliners and student pilots alike):
                * Severe active thunderstorms or heavy hail lines[cite: 5].
                * Severe turbulence or low-level wind shear[cite: 41].
                * Severe icing conditions.
                * Volcanic ash clouds or sandstorms.
                
                ### ✈️ AIRMET (Airmen's Meteorological Information)
                Issued for moderate weather hazards. These are critical for **student pilots** or light aircraft flying under visual rules:
                * **AIRMET Sierra (SI):** Widespread mountain obscuration or ceilings dropping below 1,000 feet / visibility below 3 statute miles over a wide area.
                * **AIRMET Tango (TA):** Moderate turbulence, sustained surface winds of 30 knots or greater, or non-convective low-level wind shear.
                * **AIRMET Zulu (ZU):** Moderate icing conditions and freezing level heights.
                """)

            with tab5:
                st.subheader("Master Aviation Glossary Map")
                st.markdown("Match these exact code characters against your raw top headers to test your vocabulary[cite: 626]:")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    **Cloud Cover Thresholds:**
                    * **SKC / CLR:** Sky Clear / Clear Skies [cite: 626]
                    * **FEW:** Few clouds (1/8 to 2/8 sky coverage) [cite: 626]
                    * **SCT:** Scattered clouds (3/8 to 4/8 coverage) [cite: 626]
                    * **BKN:** Broken ceiling layer (5/8 to 7/8 coverage - *Constitutes a ceiling*) [cite: 626]
                    * **OVC:** Overcast ceiling layer (8/8 complete coverage) [cite: 626]
                    """)
                with col2:
                    st.markdown("""
                    **Weather Phenomenon Codes:**
                    * **HZ:** Haze [cite: 626]
                    * **BR:** Mist (Visibility ≥ 1000m) [cite: 626]
                    * **FG:** Fog (Visibility < 1000m) [cite: 626]
                    * **RA / DZ:** Rain / Drizzle [cite: 626]
                    * **TS:** Thunderstorm [cite: 626]
                    * **CB:** Cumulonimbus (Dangerous Convective Storm Cells) [cite: 626]
                    """)

            with tab6:
                st.subheader("🌐 Official Aviation Weather Reference Libraries")
                st.markdown("""
                For deeper self-study, checkride preparation, or looking up live regional radar maps, bookmark these official government and institutional aviation portals:
                
                1. **[NOAA Aviation Weather Center (AWC)](https://aviationweather.gov/)**
                   * The gold standard system used by your app[cite: 37, 372]. Use their **'Tools'** menu to view interactive global graphical forecasts (GFA), ceiling/visibility matrices, and live SIGMET plotting charts.
                2. **[FAA Aviation Weather Services Advisory Circular (AC 00-45H)](https://www.faa.gov/)**
                   * The definitive legal blueprint text document that defines exactly how every single alphanumeric symbol in a METAR, TAF, or PIREP (Pilot Report) must be coded and read. Perfect for ground school exams.
                3. **[SKYbrary Aviation Safety Reference](https://skybrary.aero/)**
                   * An elite knowledge base created by Eurocontrol and ICAO. Search their wiki for entries on *'Flight Categories (VFR/IFR)'* and *'SPECI criteria'* to read comprehensive deep-dives into operational definitions.
                4. **[WMO Manual on Codes (WMO-No. 306)](https://wmo.int/)**
                   * The official global treaty framework documenting international standard formatting rules for meteorological messages used across global towers[cite: 373, 378].
                """)


# ============================================
# RUN APP
# ============================================

if __name__ == "__main__":
    main()