"""
Aviation Weather AI - Application Entry Point
==============================================
Launches the Streamlit application with proper
environment validation and dependency checks.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import subprocess
import logging

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / 'app.log', mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def check_environment():
    """Verify environment variables are configured"""
    from dotenv import load_dotenv
    
    env_path = PROJECT_ROOT / '.env'
    
    if not env_path.exists():
        print("\n⚠️  No .env file found!")
        print("Creating template .env file...")
        
        template = """# Aviation Weather AI - API Keys
# Get free keys from:
# Gemini: https://makersuite.google.com/app/apikey
# Groq: https://console.groq.com

GEMINI_API_KEY=your_gemini_key_here
GROQ_API_KEY=your_groq_key_here
"""
        env_path.write_text(template)
        print(f"✅ Created .env template at: {env_path}")
        print("⚠️  Edit this file with your actual API keys before running.")
        return False
    
    load_dotenv(dotenv_path=env_path)
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    warnings = []
    
    if not gemini_key or "your_gemini_key" in gemini_key:
        warnings.append("GEMINI_API_KEY not configured")
    
    if not groq_key or "your_groq_key" in groq_key:
        warnings.append("GROQ_API_KEY not configured")
    
    if warnings:
        print("\n⚠️  API Key Warnings:")
        for warning in warnings:
            print(f"   - {warning}")
        print("   The app will run but AI features may be limited.")
    
    return True


def run_app():
    """Launch the Streamlit application"""
    app_path = PROJECT_ROOT / 'app.py'
    
    if not app_path.exists():
        print(f"\n❌ app.py not found at: {app_path}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🛫  Starting Aviation Weather AI Assistant...")
    print("=" * 60)
    print(f"\n   🌐 Local:    http://localhost:8501")
    print(f"   🌐 Network:  http://0.0.0.0:8501")
    print(f"\n   📁 Project:  {PROJECT_ROOT}")
    print(f"   🐍 Python:   {sys.version.split()[0]}")
    print(f"   📅 Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n   Press Ctrl+C to stop\n")
    print("-" * 60)
    
    # Run Streamlit
    cmd = [
        sys.executable, '-m', 'streamlit', 'run',
        str(app_path),
        '--server.port=8501',
        '--server.address=0.0.0.0',
        '--browser.serverAddress=localhost',
    ]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down... Goodbye!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to start: {e}")
        print(f"\n❌ Error starting application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════╗
║  🛫  AVIATION WEATHER AI ASSISTANT  🛫     ║
║  AI-Powered METAR/TAF Briefing System       ║
║  Gemini + Groq | aviationweather.gov        ║
╚══════════════════════════════════════════════╝
    """)
    
    # Check environment
    check_environment()
    
    # Run the app
    run_app()