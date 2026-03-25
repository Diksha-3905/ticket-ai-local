"""
AI Ticket System — Local Model (No API Key Required)
Uses NLP keyword matching + TF-IDF scoring + rule-based reasoning
"""

from flask import Flask, render_template, request, jsonify
import re
from datetime import datetime

app = Flask(__name__)

# ─────────────────────────────────────────────
#  LOCAL AI MODEL — Knowledge Base
# ─────────────────────────────────────────────

KNOWLEDGE_BASE = {
    "Password & Authentication": {
        "keywords": [
            "password", "forgot", "reset", "can't log in", "cannot log in",
            "login", "log in", "sign in", "signin", "incorrect password",
            "wrong password", "locked out", "account locked", "credentials",
            "authentication", "access denied", "unable to login", "username"
        ],
        "priority_keywords": ["locked out", "can't log in", "cannot log in", "access denied"],
        "color": "#7c6aff",
        "answer": (
            "This appears to be a password or login issue. "
            "You can reset your password by visiting the login page and clicking 'Forgot Password'. "
            "A reset link will be sent to your registered email address."
        ),
        "steps": [
            "Go to the login page of your application.",
            "Click on 'Forgot Password' or 'Reset Password' link.",
            "Enter your registered email address.",
            "Check your inbox for a password reset email (also check spam folder).",
            "Click the reset link and create a new strong password.",
            "Try logging in again with your new password."
        ],
        "resolution": "5–10 minutes",
        "escalate_to": "IT Helpdesk"
    },
    "HR & Leave Management": {
        "keywords": [
            "leave", "balance", "vacation", "holiday", "time off", "pto",
            "paid leave", "sick leave", "annual leave", "leave request",
            "approve leave", "leave application", "attendance", "payroll",
            "salary", "payslip", "hr", "human resources", "days off",
            "remaining leaves", "leave policy", "casual leave"
        ],
        "priority_keywords": ["urgent leave", "emergency leave"],
        "color": "#6affcc",
        "answer": (
            "For leave balance and HR-related queries, you can check your leave balance "
            "through the HR self-service portal under 'My Leave'. "
            "Your leave balance is updated in real-time and shows all categories of leave."
        ),
        "steps": [
            "Log in to the HR Self-Service portal (HRMS).",
            "Navigate to 'My Leave' or 'Leave Management' section.",
            "Click on 'Leave Balance' to view available leaves by category.",
            "To apply for leave, click 'Apply Leave' and fill in the details.",
            "Submit the request — your manager will receive an approval notification.",
            "You'll get an email confirmation once approved or rejected."
        ],
        "resolution": "Immediate (self-service) / 1 business day (approval)",
        "escalate_to": "HR Department"
    },
    "Technical Support": {
        "keywords": [
            "error", "bug", "crash", "not working", "broken", "slow",
            "issue", "problem", "glitch", "freeze", "hang", "loading",
            "software", "application", "app", "system", "computer",
            "laptop", "device", "screen", "display", "install", "update",
            "upgrade", "download", "network", "internet", "wifi", "vpn"
        ],
        "priority_keywords": ["crash", "not working", "broken", "urgent", "critical"],
        "color": "#ffb86a",
        "answer": (
            "This looks like a technical issue. "
            "Please try basic troubleshooting steps first: restart the application or device. "
            "If the problem persists, our IT support team will assist you further."
        ),
        "steps": [
            "Close and reopen the application.",
            "Clear browser cache/cookies if it's a web app.",
            "Restart your device.",
            "Check for any pending software updates.",
            "Try accessing from a different browser or device.",
            "If issue persists, note the error message and contact IT support."
        ],
        "resolution": "30 minutes – 2 business days",
        "escalate_to": "IT Technical Support"
    },
    "Account & Access Management": {
        "keywords": [
            "account", "access", "permission", "role", "profile", "settings",
            "deactivated", "suspended", "new user", "create account",
            "onboarding", "offboarding", "transfer", "department change",
            "email change", "username change", "two factor", "2fa", "mfa"
        ],
        "priority_keywords": ["deactivated", "suspended", "no access"],
        "color": "#ff6a8a",
        "answer": (
            "This is an account or access management request. "
            "Account changes require verification and approval from your manager or IT admin. "
            "Please provide your employee ID for faster resolution."
        ),
        "steps": [
            "Confirm your identity with employee ID and department.",
            "Specify what type of access or change is needed.",
            "Get approval from your line manager if required.",
            "Submit a formal access request via the IT portal.",
            "IT Admin will process your request within 1 business day.",
            "You'll receive a confirmation email once changes are applied."
        ],
        "resolution": "1–2 business days",
        "escalate_to": "IT Admin / HR"
    },
    "General Inquiry": {
        "keywords": ["how", "what", "where", "when", "who", "why", "help", "information", "guide", "tell me"],
        "priority_keywords": [],
        "color": "#6ab8ff",
        "answer": (
            "Thank you for reaching out. "
            "Your query has been received and categorized as a general inquiry. "
            "Our support team will provide you with the relevant information shortly."
        ),
        "steps": [
            "Review the company knowledge base / FAQ section.",
            "Check if a similar query has been answered before.",
            "If not found, submit your query to the appropriate department.",
            "A support agent will respond within 1 business day."
        ],
        "resolution": "1 business day",
        "escalate_to": "General Support"
    }
}

# ─────────────────────────────────────────────
#  LOCAL AI ENGINE
# ─────────────────────────────────────────────

def tokenize(text):
    """Simple tokenizer: lowercase, remove punctuation, split."""
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    tokens = text.split()
    # Basic stopword removal
    stopwords = {"i", "me", "my", "the", "a", "an", "is", "it", "in", "to",
                 "and", "or", "but", "for", "of", "on", "at", "with", "as"}
    return [t for t in tokens if t not in stopwords]

def score_category(text_lower, tokens, category_data):
    """Score how well text matches a category using keyword overlap."""
    score = 0
    matched_keywords = []

    for kw in category_data["keywords"]:
        kw_lower = kw.lower()
        # Exact phrase match (higher weight)
        if kw_lower in text_lower:
            weight = len(kw_lower.split())  # multi-word phrases get higher weight
            score += weight * 2
            matched_keywords.append(kw)
        else:
            # Token-level match
            kw_tokens = kw_lower.split()
            if all(t in tokens for t in kw_tokens):
                score += len(kw_tokens)
                matched_keywords.append(kw)

    return score, matched_keywords

def determine_priority(text_lower, category, category_data):
    """Determine ticket priority based on urgency signals."""
    high_signals = ["urgent", "asap", "immediately", "critical", "emergency",
                    "blocked", "cannot work", "can't work", "not able to work"]
    low_signals = ["curious", "just wondering", "when you get a chance",
                   "not urgent", "low priority", "whenever"]

    # Check category-specific high priority keywords
    for kw in category_data.get("priority_keywords", []):
        if kw.lower() in text_lower:
            return "High"

    for signal in high_signals:
        if signal in text_lower:
            return "High"

    for signal in low_signals:
        if signal in text_lower:
            return "Low"

    # Default priorities per category
    defaults = {
        "Password & Authentication": "High",
        "Account & Access Management": "Medium",
        "Technical Support": "Medium",
        "HR & Leave Management": "Low",
        "General Inquiry": "Low"
    }
    return defaults.get(category, "Medium")

def calculate_confidence(score, total_tokens):
    """Convert raw score to a 0-100 confidence percentage."""
    if total_tokens == 0:
        return 0
    raw = min((score / max(total_tokens, 1)) * 100, 100)
    # Boost confidence for higher scores
    if score >= 6:
        return min(95, raw + 30)
    elif score >= 3:
        return min(85, raw + 20)
    elif score >= 1:
        return min(70, raw + 10)
    return max(30, raw)

def classify_ticket(text):
    """Main local AI classification function."""
    text_lower = text.lower().strip()
    tokens = tokenize(text)

    scores = {}
    matched = {}

    for category, data in KNOWLEDGE_BASE.items():
        score, kws = score_category(text_lower, tokens, data)
        scores[category] = score
        matched[category] = kws

    best_category = max(scores, key=scores.get)
    best_score = scores[best_category]

    # Fall back to General Inquiry if nothing matched
    if best_score == 0:
        best_category = "General Inquiry"
        best_score = 0

    cat_data = KNOWLEDGE_BASE[best_category]
    priority = determine_priority(text_lower, best_category, cat_data)
    confidence = calculate_confidence(best_score, len(tokens))

    # Build all-category scores for the breakdown chart
    all_scores = [
        {"category": cat, "score": round(calculate_confidence(sc, len(tokens)), 1)}
        for cat, sc in scores.items()
    ]
    all_scores.sort(key=lambda x: -x["score"])

    return {
        "category": best_category,
        "priority": priority,
        "confidence": round(confidence, 1),
        "answer": cat_data["answer"],
        "suggested_steps": cat_data["steps"],
        "estimated_resolution": cat_data["resolution"],
        "escalate_to": cat_data["escalate_to"],
        "matched_keywords": matched[best_category],
        "category_scores": all_scores,
        "color": cat_data["color"],
        "model": "Local Rule-Based NLP Model v1.0"
    }


# ─────────────────────────────────────────────
#  SAMPLE TICKETS
# ─────────────────────────────────────────────

SAMPLE_TICKETS = [
    {"id": 1, "text": "I forgot my password, how to reset it?"},
    {"id": 2, "text": "I can't log in, as password is incorrect."},
    {"id": 3, "text": "How to see leave balance?"},
]

# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", tickets=SAMPLE_TICKETS)

@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.json
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No ticket text provided"}), 400
    result = classify_ticket(text)
    result["ticket_text"] = text
    result["timestamp"] = datetime.now().strftime("%H:%M:%S")
    return jsonify(result)

@app.route("/api/analyze-all", methods=["POST"])
def analyze_all():
    results = []
    for ticket in SAMPLE_TICKETS:
        result = classify_ticket(ticket["text"])
        result["ticket_id"] = ticket["id"]
        result["ticket_text"] = ticket["text"]
        result["timestamp"] = datetime.now().strftime("%H:%M:%S")
        results.append(result)
    return jsonify(results)

if __name__ == "__main__":
    print("=" * 50)
    print("🎫  AI Ticket System — Local Model")
    print("    No API key required!")
    print("    Running at: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)
