## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/adityak506/Memories_In_Langchain.git
cd Memories_In_Langchain
```

### 2. Set Up a Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables ⚠️
> [!IMPORTANT]
> **API Keys and `.env` Safety:**
> The `.env` file contains sensitive API credentials and is excluded from the GitHub repository via `.gitignore` to prevent unauthorized usage or leaks.
> You **must** create this file locally on your system to run the application.

Create a file named `.env` in the root directory of the project:
```env
# For OpenAI models (if provider is set to "openai" in config.json)
OPENAI_API_KEY="your-openai-api-key-here"

# For Google Gemini models (if provider is set to "gemini" in config.json)
GOOGLE_API_KEY="your-google-api-key-here"
```

### 5. Update Configuration File
Open [`config.json`] to switch between providers (`"openai"` or `"gemini"`) and change model parameters:
```json
{
    "provider": "openai",
    "openai": {
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 200
    },
    "gemini": {
        "model": "models/gemini-3.1-flash-lite",
        "temperature": 0.7,
        "max_output_tokens": 200
    }
}
```
## 🏃 Running the Application

### Running the Terminal CLI Chatbots
Run any of the CLI scripts to test them interactively in your shell:
```bash
python 01_basic_memory.py
```
*(To exit the chat loop, type `exit` or `quit`)*

### Running the Streamlit Web App
Launch the Streamlit app to interactively switch between the different memory strategies and view how the memory contents adjust dynamically:
```bash
streamlit run 05_streamlit_memory.py
```
This will open up a browser window pointing to `http://localhost:8501`. 

---
