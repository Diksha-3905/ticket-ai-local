# 🎫 AI Ticket System — Local Model (No API Key Required)

A fully offline, rule-based NLP ticket classifier. No API key, no internet needed after setup.

---

## 🚀 Quick Start (2 Steps Only!)

### Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
```
Only **Flask** is required. That's it!

---

### Step 2 — Run the App
```bash
python app.py
```
Open your browser: **http://localhost:5000**

---

## 🧠 How the Local AI Model Works

### Architecture: Rule-Based NLP Engine

```
Input Ticket Text
       │
       ▼
  Tokenizer  ──── Lowercase + Remove punctuation + Split words
       │
       ▼
 Keyword Scorer ── TF-IDF-style keyword matching against Knowledge Base
       │
       ▼
 Category Picker ── Best-matching category wins
       │
       ▼
Priority Engine ── Urgency signal detection (locked out = High, etc.)
       │
       ▼
 Answer Builder ── Returns pre-built answer + steps for that category
```

### Categories & Keywords

| Category | Trigger Keywords |
|---|---|
| **Password & Authentication** | password, forgot, reset, can't log in, login, incorrect, locked out... |
| **HR & Leave Management** | leave, balance, vacation, pto, payroll, attendance... |
| **Technical Support** | error, crash, bug, slow, not working, software, network... |
| **Account & Access Management** | account, permission, role, deactivated, 2fa, onboarding... |
| **General Inquiry** | how, what, where, help, information... |

### Confidence Score
- Calculated from keyword match density vs token count
- Multi-word phrase matches score higher (e.g. "can't log in" > "login")
- Score is boosted for stronger matches

### Priority Logic
- **High** → "locked out", "can't log in", "urgent", "blocked", "critical"
- **Medium** → Default for technical and account issues
- **Low** → Informational queries ("how to", "what is")

---

## 📁 File Structure

```
ticket-ai-local/
├── app.py              ← Flask app + complete local AI model
├── requirements.txt    ← Only 'flask' needed
├── README.md           ← This file
└── templates/
    └── index.html      ← Full UI with confidence meter + grouping
```

---

## ✨ Features

- ✅ **Zero API key** — runs 100% offline
- ✅ **Only Flask needed** — no heavy ML libraries
- ✅ AI confidence score with animated bar
- ✅ Keyword match highlighting
- ✅ Ticket grouping by category
- ✅ Step-by-step resolution guide
- ✅ Priority tagging (High / Medium / Low)
- ✅ Custom ticket input
- ✅ Beautiful UI with dark/light polish

---

## 🔧 Troubleshooting

**Port in use?** → Edit last line of `app.py`: `app.run(port=5001)`

**ModuleNotFoundError** → Run `pip install flask`
