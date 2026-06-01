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

from src.fetcher import fetch_metar, fetch_taf
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
        "raw_metar": "",
        "raw_taf": "",
        "search_history": [],
        "audio_generated": False,
        "audio_data": None,
        "show_educational": False,
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
        st.caption("AI-Powered Pilot Briefing System")
        
        st.divider()
        
        # AI Status Section
        st.markdown("#### 🤖 AI Engine Status")
        ai_status = get_ai_status()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if ai_status.get("deepseek_available"):
                st.success("DSeek ✅")
            else:
                st.error("DSeek ❌")
        
        with col2:
            if ai_status.get("gemini_available"):
                st.success("Gemini ✅")
            else:
                st.error("Gemini ❌")
        
        with col3:
            if ai_status.get("groq_available"):
                st.success("Groq ✅")
            else:
                st.error("Groq ❌")
        
        # Rate limit info
        if ai_status.get("rate_limits"):
            stats = ai_status["rate_limits"]
            st.caption(f"Minute calls: {stats.get('calls_this_minute', {})}")
            st.caption(f"Reset in: {stats.get('seconds_until_reset', 0)}s")
            
        # Cache & Queue Stats
        cache_stats = ai_status.get("cache_stats", {"memory_entries": 0, "disk_entries": 0})
        st.caption(f"📦 Cache: {cache_stats.get('memory_entries', 0)} memory / {cache_stats.get('disk_entries', 0)} disk entries")
        
        queue_stats = ai_status.get("queue_stats", {"queue_length": 0, "currently_processing": 0, "max_concurrent": 5})
        q_len = queue_stats.get("queue_length", 0)
        q_proc = queue_stats.get("currently_processing", 0)
        q_max = queue_stats.get("max_concurrent", 5)
        
        if q_len > 0:
            st.warning(f"⏳ Queue: {q_len} waiting ({q_proc}/{q_max} active)")
        else:
            st.caption(f"🟢 Queue: 0 waiting ({q_proc}/{q_max} active)")
            
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
        
        # Display Options
        st.markdown("#### ⚙️ Display Options")
        st.session_state.show_educational = st.toggle(
            "📚 Show Educational Insights",
            value=st.session_state.show_educational
        )
        
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
            st.metric("🤖 AI Models", "3")
            
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
        st.caption("AI by Gemini & Groq (Free Tier)")
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
    
    # AI Analysis
    with st.spinner("🤖 AI analyzing weather patterns (will queue if busy)..."):
        try:
            # Directly call orchestrator (which returns a dict and handles caching & queuing)
            analysis = analyze_metar_orchestrator(raw_metar, raw_taf)
            
            # Format for display
            analysis = format_analysis_for_display(analysis)
            
            # Store in session state
            st.session_state.analysis = analysis
            st.session_state.station_loaded = icao
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
            elif analysis.get("fallback_used") and not analysis.get("ai_available", True):
                st.toast("🚦 High traffic: Basic parsing fallback used.", icon="⚠️")
            else:
                st.toast("✅ Full AI Weather Briefing Generated!", icon="🤖")
                
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


def display_educational_insights(analysis: dict):
    """Display educational insights for student pilots"""
    
    if not st.session_state.show_educational:
        return
    
    insights = analysis.get('educational_insights', '')
    if not insights or insights.startswith('⚠️ TEACHING MODE OFFLINE'):
        return
    
    st.markdown('<p class="section-title">📚 Student Pilot Learning Corner</p>', unsafe_allow_html=True)
    
    with st.expander("🎓 View Educational Insights", expanded=True):
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); 
                    padding: 20px; border-radius: 10px; border: 1px solid #667eea30;">
            {insights}
        </div>
        """, unsafe_allow_html=True)
        
        # Add quiz interaction
        if "QUESTION:" in insights.upper():
            if st.button("📝 Reveal Answer", key="reveal_answer"):
                # Find the answer part
                answer_start = insights.upper().find("ANSWER:")
                if answer_start > 0:
                    answer = insights[answer_start:]
                    st.success(answer)
                else:
                    st.info("Answer embedded in the insights above")


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
    provider = analysis.get("provider", "unknown")
    fallback = analysis.get("fallback_used", False)
    
    provider_color = {
        "deepseek": "#0066FF",
        "gemini": "#4285F4",
        "groq": "#F50057",
        "regex_parser": "#FF9800"
    }
    
    st.markdown(f"""
    <span style="background: {provider_color.get(provider, '#gray')}15; 
                 color: {provider_color.get(provider, '#gray')}; 
                 padding: 4px 12px; 
                 border-radius: 12px; 
                 font-size: 0.8rem;">
        🤖 AI: {provider.upper()} {'(Fallback)' if fallback else ''}
    </span>
    """, unsafe_allow_html=True)
    
    if analysis.get("status") == "partial":
        st.info("ℹ️ Some fields may be incomplete")
    
    # Get the actual analysis data
    analysis_data = analysis.get("analysis", analysis)
    
    # Success message
    st.success(f"✅ Weather Analysis Complete for {analysis_data.get('station', 'Unknown')}")
    
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
    
    # Educational Insights (if enabled)
    display_educational_insights(analysis_data)
    
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
    
    with streamlit_analytics.track(data_file="weather_analytics.json"):
        # Render sidebar
        render_sidebar()
        
        # Main content area
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.markdown('<h1 class="main-title">🛫 ATC Weather Assistant</h1>', unsafe_allow_html=True)
            st.caption("AI-Powered Structural Pilot Briefing Sequencer")
        
        st.divider()
        
        # Check AI availability before analysis
        ai_status = get_ai_status()
        is_overloaded = not ai_status.get('gemini_available') and not ai_status.get('groq_available') and not ai_status.get('deepseek_available')
        if is_overloaded:
            show_high_traffic_mode()
        
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
                ### 🤖 AI Analysis
                Dual-engine AI (Gemini + Groq) decodes METAR/TAF into plain English with 
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


# ============================================
# RUN APP
# ============================================

if __name__ == "__main__":
    main()