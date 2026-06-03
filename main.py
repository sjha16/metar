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
    """Verify environment variables are loaded"""
    from dotenv import load_dotenv
    
    env_path = PROJECT_ROOT / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    return True


def run_app():
    """Launch the Streamlit application"""
    app_path = PROJECT_ROOT / 'app.py'
    
    if not app_path.exists():
        print(f"\n❌ app.py not found at: {app_path}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🛫  Starting ATC Weather Assistant...")
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
║  🛫  ATC METAR & TAF WEATHER ASSISTANT 🛫     ║
║  Automated METAR/TAF Briefing System         ║
║  Regex-Powered Parser Only                   ║
╚══════════════════════════════════════════════╝
    """)
    
    # Check environment
    check_environment()
    
    # Run the app
    run_app()