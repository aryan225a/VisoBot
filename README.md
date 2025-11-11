# 🔷 VisoBot - AI Object Detection & Assistant

VisoBot is a real-time object detection and tracking application with an integrated AI assistant. It uses a webcam feed to identify objects, tracks them across frames, and allows users to ask questions about the detected objects via Google's Gemini AI. This project combines the power of computer vision (YOLOv8) with natural language processing (Gemini) to create an interactive and educational tool.


## 🚀 Key Features

* **Real-time Object Detection:** Identifies 80 common object classes from the COCO dataset using the lightweight YOLOv8n model.
* **Persistent Object Tracking:** Implements DeepSORT to assign and maintain unique IDs for objects, tracking them smoothly across frames even with temporary obstructions.
* **🤖 AI-Powered Q&A:** Integrates with the Gemini 2.0 Flash model, allowing users to select a detected object and ask natural language questions about it (e.g., "What are its nutritional benefits?").
* **Interactive Web UI:** Built with Streamlit, providing a clean, user-friendly, and responsive interface that runs in the browser.
* **Performance Optimized:**
    * **Threaded Video Capture:** Uses a non-blocking, threaded approach (`VideoGet` class) to read webcam frames, reducing lag and jitter.
    * **Frame Skipping:** Users can configure the app to process every Nth frame, balancing CPU load and detection frequency.
    * **API Caching:** AI responses from Gemini are cached (for 1 hour) to reduce API calls and provide faster answers for repeated queries.
* **Customizable Controls:** Users can adjust the **Confidence Threshold** and **Frame Processing Rate** directly from the sidebar to fine-tune performance.

---

## 🛠️ Tech Stack

* **Core:** Python 3.10+
* **Web Framework:** Streamlit
* **Object Detection:** Ultralytics YOLOv8n
* **Object Tracking:** DeepSORT (`deep_sort_realtime`)
* **AI Assistant:** Google Gemini (via `google.genai`)
* **CV & Numerics:** OpenCV (`cv2`), NumPy
* **Configuration:** `python-dotenv`

---

## 📦 Installation & Setup

Follow these steps to get VisoBot running on your local machine.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/aryan225a/VisoBot](https://github.com/aryan225a/VisoBot)
    cd visobot
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv visobot_env
    source visobot_env/bin/activate # On Windows: visobot_env\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install opencv-python streamlit ultralytics deep-sort-realtime google-generativeai python-dotenv numpy
    ```

4.  **Set up your environment variables:**
    Create a file named `.env` in the root directory. You will need to add your Google Gemini API key to this file. You can get a free API key from [Google AI Studio](https://ai.google.dev/).

    ```
    GOOGLE_API_KEY=your_api_key_here
    ```

---

## ▶️ How to Run

Once everything is installed and your `.env` file is configured, run the Streamlit app from your terminal:

```bash
streamlit run visobot.py
```

The application will automatically open in your default web browser. You may need to grant browser and OS permissions for the app to access your webcam.