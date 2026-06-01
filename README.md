# 🛫 ATC METAR & TAF Weather Briefing Assistant

An advanced, AI-powered weather decoding and safety-assessment system built specifically for pilots and flight instructors. This application takes raw meteorological reports (METAR) and terminal aerodrome forecasts (TAF) and translates them into plain English, provides student pilot safety evaluations, delivers voice briefings, and serves interactive educational breakdowns.

---

## ✨ Key Features

* **🤖 Dual-Engine AI Decoders**: Utilizes Google Gemini 2.5 Flash as the primary processor with a robust Groq (Llama 3.3) fallback to decode raw weather parameters.
* **🛡️ Smart Personal Minimums Evaluator**: Evaluates current and forecast conditions against strict student pilot personal minimums (crosswinds, ceilings, visibility, convective hazards) to issue a definitive **GO / NO-GO** safety recommendation.
* **🎙️ Voice Briefing Synthesizer**: Generates professional hands-free audio briefings of the meteorological report using Text-to-Speech (`gTTS`).
* **📚 Student Pilot Learning Corner**: Provides comprehensive, line-by-line interactive explanations of raw aviation abbreviations and includes a dynamic checkride oral exam simulator.
* **⚡ Quick Access Buttons**: Instant live weather retrieval for key Indian and international airports.
* **🔌 Robust Local Fallbacks**: Equipped with an automated, regex-based offline parsing mode in case of API rate limits or network issues.

---

## 🛠️ Technology Stack

* **Frontend**: [Streamlit](https://streamlit.io/) (High-performance web dashboard)
* **AI & Language Processing**: 
  * Primary: [Google GenAI SDK](https://github.com/google/generative-ai-python) (Gemini 2.5)
  * Fallback: [Groq SDK](https://github.com/groq/groq-python) (Llama 3.3)
  * Schema Validation: [Pydantic v2](https://docs.pydantic.dev/)
* **Audio Synthesis**: [gTTS](https://github.com/pndurette/gTTS) (Google Text-to-Speech)
* **Data Retrieval**: Live weather reports directly from [AviationWeather.gov](https://aviationweather.gov/)

---

## 🚀 Local Quickstart

### 1. Clone & Initialize Environment
Clone the repository and create a Python virtual environment:
```bash
git clone <your-repository-url>
cd metar
python -m venv .venv
```

Activate the virtual environment:
* **Windows (PowerShell)**: `.venv\Scripts\Activate.ps1`
* **macOS / Linux**: `source .venv/bin/activate`

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Credentials
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY="your_google_gemini_api_key"
GROQ_API_KEY="your_groq_api_key_here"
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser!

---

## ☁️ Deployment

This project is fully optimized for **Streamlit Community Cloud**. 
For step-by-step instructions on setting up GitHub sync and adding production credentials, please refer to the [Streamlit Cloud Deployment Guide](C:\Users\Sumit\.gemini\antigravity\brain\f98c2c82-98e0-4012-8b84-c7c44f4ecf2e\deployment_guide.md).
