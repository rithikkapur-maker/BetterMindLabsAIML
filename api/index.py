import os
import json
import re
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

base_dir = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(base_dir, '../templates'))

# Ensure JSON outputs maintain true Unicode characters (e.g. Greek symbols) instead of escaped ASCII
app.json.ensure_ascii = False

# Initialize high-speed Groq Inference Engine client using the 2026 flagship open model
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


def _to_float(value, fallback):
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def sanitize_interactive_graph(graph):
    """Normalize the AI-generated graph spec so a non-compliant response
    can never reach the frontend and silently break the plot: the expression
    must genuinely vary with x, 1-2 numeric sliders that each actually affect
    the curve, and sane axis bounds."""
    if not isinstance(graph, dict):
        return None

    expression = graph.get("expression")
    if not isinstance(expression, str) or not expression.strip():
        return None

    # The curve must actually respond to the x-axis. If the expression never
    # references 'x', moving along the x-axis has zero effect on y - a flat,
    # broken-looking line - so reject it outright rather than render it.
    if not re.search(r'\bx\b', expression):
        return None

    raw_sliders = graph.get("sliders")
    if not isinstance(raw_sliders, list):
        return None

    sliders = []
    for s in raw_sliders:
        if not isinstance(s, dict):
            continue
        variable = s.get("variable")
        if not isinstance(variable, str) or not variable.strip() or variable == "x":
            continue
        # Only keep sliders that actually influence the plotted expression.
        if not re.search(r'\b' + re.escape(variable) + r'\b', expression):
            continue
        s_min = _to_float(s.get("min"), 0)
        s_max = _to_float(s.get("max"), s_min + 10)
        if s_max <= s_min:
            s_max = s_min + 10
        s_value = _to_float(s.get("value"), (s_min + s_max) / 2)
        s_value = min(max(s_value, s_min), s_max)
        s_step = _to_float(s.get("step"), 0.5)
        if s_step <= 0:
            s_step = 0.5
        sliders.append({
            "variable": variable,
            "label": s.get("label") or variable,
            "min": s_min,
            "max": s_max,
            "value": s_value,
            "step": s_step,
        })
        if len(sliders) == 2:
            break

    # The feature is "1-2 fields the student can manipulate that each visibly
    # change the curve" - zero valid sliders means there's nothing to interact
    # with, so drop the graph rather than render a static, broken one.
    if len(sliders) == 0:
        return None

    x_axis = graph.get("xAxis") if isinstance(graph.get("xAxis"), dict) else {}
    y_axis = graph.get("yAxis") if isinstance(graph.get("yAxis"), dict) else {}

    x_min = _to_float(x_axis.get("min"), 0)
    x_max = _to_float(x_axis.get("max"), x_min + 10)
    if x_max <= x_min:
        x_max = x_min + 10
    x_step = _to_float(x_axis.get("step"), (x_max - x_min) / 100 or 0.1)
    if x_step <= 0:
        x_step = (x_max - x_min) / 100 or 0.1

    y_min = _to_float(y_axis.get("min"), -10)
    y_max = _to_float(y_axis.get("max"), 10)
    if y_max <= y_min:
        y_min, y_max = -10, 10

    return {
        "title": graph.get("title") or "Interactive Concept Simulation",
        "sliders": sliders,
        "xAxis": {"label": x_axis.get("label") or "x", "min": x_min, "max": x_max, "step": x_step},
        "yAxis": {"label": y_axis.get("label") or "y", "min": y_min, "max": y_max},
        "expression": expression,
    }


@app.route('/')
def home():
    supabase_url = os.environ.get("SUPABASE_URL", "")
    supabase_anon_key = os.environ.get("SUPABASE_ANON_KEY", "")
    return render_template('index.html', supabase_url=supabase_url, supabase_anon_key=supabase_anon_key)


@app.route('/dashboard')
def dashboard():
    supabase_url = os.environ.get("SUPABASE_URL", "")
    supabase_anon_key = os.environ.get("SUPABASE_ANON_KEY", "")
    return render_template('index.html', supabase_url=supabase_url, supabase_anon_key=supabase_anon_key)


@app.route('/api/morph-concept', methods=['POST'])
def morph_concept():
    if not client:
        return jsonify({"error": "Groq API Key configuration missing from deployment runtime environment."}), 500

    data = request.get_json() or {}
    topic = data.get('topic', '').strip()
    profile = data.get('profile', 'ELI5').strip()
    depth = data.get('depth', 'Undergrad').strip()

    if not topic:
        return jsonify({"error": "Target concept query value cannot be empty."}), 400

    system_prompt = (
        "You are an elite AI cognitive adaptation solution architect optimized for US educational paradigms.\n"
        "Your task is to disassemble complex technical topics and reconstitute them completely aligned to the user's chosen conceptual lens profile and depth level.\n\n"
        "You MUST return your response exclusively as a valid JSON object matching this schema blueprint:\n"
        "{\n"
        "  \"morphed_explanation\": \"A multi-paragraph highly personalized conceptual explanation mapping analogies from the selected profile lens tailored for the target academic depth.\",\n"
        "  \"takeaways\": [\"First bulletproof core essential pillar\", \"Second bulletproof core essential pillar\", \"Third bulletproof core essential pillar\"],\n"
        "  \"interactive_graph\": {\n"
        "    \"title\": \"Dynamic Graph Title (e.g., Simple Harmonic Motion displacement)\",\n"
        "    \"sliders\": [\n"
        "      { \"variable\": \"A\", \"label\": \"Amplitude (A)\", \"min\": 1, \"max\": 10, \"value\": 5, \"step\": 0.5 },\n"
        "      { \"variable\": \"w\", \"label\": \"Angular Frequency (w)\", \"min\": 1, \"max\": 10, \"value\": 3, \"step\": 0.5 }\n"
        "    ],\n"
        "    \"xAxis\": { \"label\": \"Time (s)\", \"min\": 0, \"max\": 10, \"step\": 0.1 },\n"
        "    \"yAxis\": { \"label\": \"Displacement (m)\", \"min\": -10, \"max\": 10 },\n"
        "    \"expression\": \"A * Math.sin(w * x)\"\n"
        "  }\n"
        "}\n\n"
        "CRITICAL RULES FOR 'interactive_graph':\n"
        "1. If the topic involves quantitative equations, mathematical relations, science calculations, physics, or graphable quantitative formulas, you MUST populate 'interactive_graph'.\n"
        "2. FIELD SELECTION (MOST IMPORTANT RULE): The x-axis of the graph is ALWAYS one continuously-varying independent quantity for the topic (e.g. Time, Displacement, Acceleration). Beyond that x-axis quantity, you MUST choose the 1 or 2 most pedagogically essential REMAINING quantities as sliders — whichever number is mathematically correct for that specific equation, never a fixed count. Prefer 2 sliders whenever the equation genuinely has 2 independent coefficients distinct from the x-axis (e.g., SHM: Amplitude and Angular Frequency, with Time as x-axis; a loan/interest topic: Principal and Interest Rate, with Time as x-axis). Use only 1 slider when the equation is a simple two-variable product/relation where the OTHER variable IS the x-axis (e.g., Newton's Second Law F = m * a: if Acceleration is the x-axis, then Mass is the only slider — do NOT also add a redundant 'Acceleration' slider, since that would duplicate the x-axis and produce a flat, unresponsive line). NEVER create a slider whose physical meaning duplicates the x-axis quantity. A student must be able to move every slider and immediately see a distinct, understandable change in the curve.\n"
        "3. The 'expression' MUST be a valid mathematical expression in standard JavaScript syntax that can be evaluated dynamically using variables (e.g., 'A * Math.sin(w * x)'). Use standard JS functions like Math.sin, Math.cos, Math.pow, Math.exp, Math.sqrt, etc.\n"
        "4. Keep slider values, axis bounds, and steps reasonable and positive unless negative values are required.\n"
        "5. DERIVATIVE / ACCUMULATION RULE: If the topic is velocity, acceleration, power, or another rate concept:\n"
        "   - Plot the ACCUMULATED quantity on the Y-axis (e.g., Displacement for velocity) against Time on the X-axis.\n"
        "   - Provide sliders for the rate (e.g., 'v' for Velocity) and initial value (e.g., 's0' for Initial Displacement).\n"
        "6. If the topic is purely qualitative or descriptive (e.g., history, literature, legal definitions), you MUST set 'interactive_graph' to null.\n"
        "7. CANONICAL FORM FOR BROAD/GENERIC TOPICS: If the topic is a single broad physical quantity, plot the quintessential defining formula.\n\n"
        "Do not wrap your response in markdown text wrappers or include backticks. Return the raw JSON object directly."
    )

    user_prompt = f"""
    Target Academic Concept: {topic}
    Cognitive Personalization Profile Lens: {profile}
    US Academic Depth Tier Setting: {depth}

    Processing Rules:
    1. Reframe explanation paradigms cleanly using the style rules of the '{profile}' model.
    2. Enforce structural depth and syntax criteria normalized to US {depth} educational requirements.
    3. Formulate exactly 3 bulletproof architectural takeaways detailing structural pillars.
    """

    try:
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="openai/gpt-oss-20b",
            temperature=0.3,
            max_tokens=4096,
            extra_body={"reasoning_effort": "low"},
            response_format={"type": "json_object"}
        )

        response_data = json.loads(completion.choices[0].message.content)
        if "interactive_graph" in response_data:
            response_data["interactive_graph"] = sanitize_interactive_graph(response_data["interactive_graph"])
        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/generate-evaluation', methods=['POST'])
def generate_evaluation():
    if not client:
        return jsonify({"error": "Groq API Key configuration missing from deployment runtime environment."}), 500

    data = request.get_json() or {}
    topic = data.get('topic', '').strip()
    eval_type = data.get('type', 'Quiz').strip()
    difficulty = data.get('difficulty', 'Intermediate').strip()
    gap_history = data.get('gap_history', [])

    if not topic:
        return jsonify({"error": "Target concept query value cannot be empty."}), 400

    history_injection = ""
    if gap_history and isinstance(gap_history, list):
        history_str = "; ".join([str(g) for g in gap_history[:5]])
        history_injection = f"Target these identified knowledge gaps in the user's prior evaluation attempts: {history_str}"

    system_prompt = (
        "You are an adaptive educational assessment engine.\n"
        "Generate a set of evaluation items (flashcards or multiple choice questions) based on the topic and difficulty level.\n"
        "You MUST return your response exclusively as a JSON object matching this schema:\n"
        "{\n"
        "  \"items\": [\n"
        "    {\n"
        "      \"question\": \"The question or flashcard front text\",\n"
        "      \"options\": [\"Option A\", \"Option B\", \"Option C\", \"Option D\"],\n"
        "      \"correct_index\": 0,\n"
        "      \"explanation\": \"Granular diagnostic feedback explicitly explaining why the choice is true and why alternatives fail\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "For Flashcards, 'options' can be omitted or empty, and 'explanation' acts as the back of the flashcard.\n"
        "Do not wrap your response in markdown text wrappers or include backticks. Return the raw JSON object directly."
    )

    user_prompt = f"""
    Testing Objective Concept: {topic}
    Evaluation System Type: {eval_type}
    Difficulty Tier Target: {difficulty}
    {history_injection}

    Execution Matrix Requirements:
    1. Generate exactly 4 distinct highly-targeted evaluation items.
    2. Match structural depth definitions expected by US grading standards for the {difficulty} tier.
    3. Provide definitive analytical reasoning answers for real-time validation checks.
    """

    try:
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="openai/gpt-oss-20b",
            temperature=0.4,
            max_tokens=4096,
            extra_body={"reasoning_effort": "low"},
            response_format={"type": "json_object"}
        )

        response_data = json.loads(completion.choices[0].message.content)
        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/generate-research', methods=['POST'])
def generate_research():
    if not client:
        return jsonify({"error": "Groq API Key configuration missing from deployment runtime environment."}), 500

    data = request.get_json() or {}
    topic = data.get('topic', '').strip()
    profile = data.get('profile', 'ELI5').strip()
    depth = data.get('depth', 'Undergrad').strip()

    if not topic:
        return jsonify({"error": "Target concept query value cannot be empty."}), 400

    system_prompt = (
        "You are an executive research synthesis AI.\n"
        "Generate a structured research brief on the given topic formatted as a JSON object with this exact structure:\n"
        "{\n"
        "  \"title\": \"Title of Research Brief\",\n"
        "  \"abstract\": \"Executive summary of the research topic...\",\n"
        "  \"sections\": [\n"
        "    {\n"
        "      \"heading\": \"Section Title\",\n"
        "      \"content\": \"Detailed section content with inline citation tags like [1] or [2]...\"\n"
        "    }\n"
        "  ],\n"
        "  \"references\": [\n"
        "    { \"key\": 1, \"text\": \"Full canonical scientific citation (Author, Year. 'Title.' Journal/Publisher or Government Database Registry)\" }\n"
        "  ],\n"
        "  \"followup_questions\": [\n"
        "    \"First deeply analytical follow-up question to prompt further study or critical inquiry\",\n"
        "    \"Second deeply analytical follow-up question to prompt further study or critical inquiry\",\n"
        "    \"Third deeply analytical follow-up question to prompt further study or critical inquiry\"\n"
        "  ]\n"
        "}\n\n"
        "Do not wrap your response in markdown text wrappers or include backticks. Return the raw JSON object directly."
    )

    user_prompt = f"""
    Research Subject Matter: {topic}
    Cognitive Lens Profile: {profile}
    US Academic Depth Tier: {depth}

    Processing Rules:
    1. Reframe explanation paradigms cleanly using the style rules of the '{profile}' model.
    2. Enforce structural depth and syntax criteria normalized to US {depth} educational requirements.
    3. Ensure inline citation tags [1], [2], etc. cleanly link to realistic, high-quality sources in the 'references' array.
    4. Generate exactly 3 highly stimulating analytical follow-up questions to prompt the user for further exploration.
    """

    try:
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="openai/gpt-oss-20b",
            temperature=0.4,
            max_tokens=4096,
            extra_body={"reasoning_effort": "low"},
            response_format={"type": "json_object"}
        )

        response_data = json.loads(completion.choices[0].message.content)
        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)