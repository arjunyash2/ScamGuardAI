# 🛡️ ScamGuardAI

**AI-powered Fraud & Spam Email Classifier with LLM-based Risk Scoring**

---

## 📖 Overview

**ScamGuardAI** is an intelligent email fraud detection system that classifies messages as *safe*, *suspicious*, or *spam*, and explains the reasoning behind its decision using a Large Language Model (LLM).

It combines **machine learning (ML)** signals with **LLM-based reasoning** to deliver both accuracy and explainability — ideal for users, organizations, and researchers who want transparent AI-driven email analysis.

---

## 🎯 Features

✅ **Email Classification:** Detects spam, phishing, and fraudulent patterns in email text.

✅ **LLM-based Risk Scoring:** Generates a 0-100 “risk score” with a short rationale for interpretability.

✅ **Batch Processing:** Analyze multiple emails simultaneously via the app interface.

✅ **Modern UI:** Interactive dashboard built with Streamlit for visualization and results.

✅ **Explainable AI:** Combines statistical classification with LLM judgment prompts for human-readable risk explanations.

✅ **Secure Local Execution:** Works fully offline — no data leaves your system.

---

## 🧱 Tech Stack

| Layer             | Tools                                      |
| ----------------- | ------------------------------------------ |
| **Language**      | Python 3.9+                                |
| **Libraries**     | TensorFlow / Scikit-Learn / Pandas / NumPy |
| **LLM Layer**     | LangChain + Local LLaMA 2 (via Ollama)     |
| **Frontend**      | Streamlit                                  |
| **Visualization** | Matplotlib / Plotly                        |
| **Deployment**    | Streamlit / Chrome Extension Prototype     |

---

## 🧠 How It Works

1. **Data Preprocessing:** Extract email bodies, remove headers, tokenize, and clean text.
2. **ML Classification:** Predict base spam/safe probability using trained model (Naive Bayes / TensorFlow).
3. **LLM Reasoning:** Send the email content and ML results to an LLM via LangChain for qualitative judgment.
4. **Risk Scoring:** LLM outputs a structured JSON with `risk_score` and `reason`.
5. **Visualization:** Streamlit displays progress bars, color-coded labels, and explanations.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/arjunyash2/ScamGuardAI.git
cd ScamGuardAI
```

### 2️⃣ Create a virtual environment

```bash
python -m venv tf-env
source tf-env/bin/activate     # macOS/Linux
tf-env\Scripts\activate        # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the app

```bash
streamlit run app.py
```

Then open: 👉 **[http://localhost:8501](http://localhost:8501)**

---

## 🧩 Example Output

| Email                                    | Risk Score | Classification | LLM Explanation                     |
| ---------------------------------------- | ---------- | -------------- | ----------------------------------- |
| “You’ve won $10,000! Click to claim!”    | 96         | 🟥 High Risk   | Contains urgency + suspicious link  |
| “Invoice attached for your subscription” | 43         | 🟧 Medium      | No personal info, but link mismatch |
| “Meeting rescheduled to 3 PM”            | 5          | 🟩 Safe        | Legitimate context and no red flags |

---

## 🧮 Model Architecture

```
        ┌──────────────────────────┐
        │      Email Text Input     │
        └────────────┬─────────────┘
                     │
          ┌──────────▼──────────┐
          │   Preprocessing     │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │   ML Classifier     │  → Base Probability
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │   LLM (LangChain)   │  → Risk Score + Reason
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │   Streamlit UI      │
          └─────────────────────┘
```

---

## 📈 Future Enhancements

* 🔹 Chrome extension that flags risky emails directly in Gmail UI.
* 🔹 Integration with Outlook / Office365 APIs.
* 🔹 Dataset expansion with multilingual email samples.
* 🔹 Model fine-tuning using OpenAI’s or LLaMA’s instruction datasets.
* 🔹 Enterprise dashboard for SOC (Security Operations Center) teams.
