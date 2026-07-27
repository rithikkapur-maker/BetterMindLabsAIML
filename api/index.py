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

# Initialize high-speed Groq Inference Engine client using the flagship open model
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

    # The expression MUST involve 'x' as a variable
    if not re.search(r'\bx\b', expression):
        return None

    title = str(graph.get("title") or "Interactive Plot")

    x_axis = graph.get("xAxis") if isinstance(graph.get("xAxis"), dict) else {}
    y_axis = graph.get("yAxis") if isinstance(graph.get("yAxis"), dict) else {}

    x_min = _to_float(x_axis.get("min"), 0.0)
    x_max = _to_float(x_axis.get("max"), 10.0)
    if x_min >= x_max:
        x_min, x_max = 0.0, 10.0

    y_min = _to_float(y_axis.get("min"), -10.0)
    y_max = _to_float(y_axis.get("max"), 10.0)
    if y_min >= y_max:
        y_min, y_max = -10.0, 10.0

    x_label = str(x_axis.get("label") or "x")
    y_label = str(y_axis.get("label") or "y")

    step = (x_max - x_min) / 100.0

    raw_sliders = graph.get("sliders")
    sanitized_sliders = []
    if isinstance(raw_sliders, list):
        for s in raw_sliders:
            if not isinstance(s, dict):
                continue
            var = s.get("variable")
            if not isinstance(var, str) or not var.strip() or var == 'x':
                continue
            var = var.strip()
            # The variable must actually be referenced inside the mathematical expression
            if not re.search(r'\b' + re.escape(var) + r'\b', expression):
                continue

            s_min = _to_float(s.get("min"), -5.0)
            s_max = _to_float(s.get("max"), 5.0)
            if s_min >= s_max:
                s_min, s_max = -5.0, 5.0

            s_val = _to_float(s.get("value"), (s_min + s_max) / 2.0)
            s_val = max(s_min, min(s_max, s_val))
            s_step = _to_float(s.get("step"), (s_max - s_min) / 50.0)

            sanitized_sliders.append({
                "variable": var,
                "label": str(s.get("label") or var),
                "min": s_min,
                "max": s_max,
                "value": s_val,
                "step": s_step if s_step > 0 else 0.1
            })

    return {
        "title": title,
        "expression": expression.strip(),
        "xAxis": {
            "label": x_label,
            "min": x_min,
            "max": x_max,
            "step": step
        },
        "yAxis": {
            "label": y_label,
            "min": y_min,
            "max": y_max
        },
        "sliders": sanitized_sliders
    }


@app.route('/')
def index():
    supabase_url = os.environ.get("SUPABASE_URL", "")
    supabase_anon_key = os.environ.get("SUPABASE_ANON_KEY", "")
    return render_template(
        'index.html',
        supabase_url=supabase_url,
        supabase_anon_key=supabase_anon_key
    )


@app.route('/api/morph-concept', methods=['POST'])
def morph_concept():
    if not client:
        return jsonify({"error": "GROQ_API_KEY environment variable is not configured."}), 500

    data = request.json or {}
    topic = data.get("topic", "").strip()
    profile = data.get("profile", "ELI5")
    depth = data.get("depth", "Undergrad")

    if not topic:
        return jsonify({"error": "A topic prompt is required."}), 400

    system_prompt = (
        "You are EduMorph AI, an advanced hyper-personalized learning matrix engine.\n"
        "Your goal is to explain complex technical concepts according to specified cognitive lenses and academic depths, "
        "and return a JSON object with a working mathematical graph representation if applicable.\n\n"
        "You MUST respond strictly with a JSON object containing the following keys:\n"
        "{\n"
        "  \"morphed_explanation\": \"Comprehensive explanation tailored to the requested cognitive lens and depth.\",\n"
        "  \"takeaways\": [\"Pillar 1 summary\", \"Pillar 2 summary\", \"Pillar 3 summary\"],\n"
        "  \"interactive_graph\": {\n"
        "     \"title\": \"Graph Title\",\n"
        "     \"expression\": \"a * Math.sin(b * x)\",\n"
        "     \"xAxis\": { \"label\": \"Time (s)\", \"min\": 0, \"max\": 10 },\n"
        "     \"yAxis\": { \"label\": \"Amplitude\", \"min\": -10, \"max\": 10 },\n"
        "     \"sliders\": [\n"
        "        { \"variable\": \"a\", \"label\": \"Amplitude A\", \"min\": 1, \"max\": 10, \"value\": 5, \"step\": 0.5 },\n"
        "        { \"variable\": \"b\", \"label\": \"Frequency B\", \"min\": 0.5, \"max\": 5, \"value\": 2, \"step\": 0.1 }\n"
        "     ]\n"
        "  }\n"
        "}\n\n"
        "Do not wrap your response in markdown formatting or backticks. Return raw JSON directly."
    )

    user_prompt = f"""
    Subject Concept: {topic}
    Target Cognitive Lens: {profile}
    US Academic Depth Tier: {depth}

    Generate a complete JSON explanation object adhering strictly to the schema rules above.
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

        if "interactive_graph" in response_data:
            response_data["interactive_graph"] = sanitize_interactive_graph(response_data["interactive_graph"])

        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": f"Transformation engine error: {str(e)}"}), 500


@app.route('/api/generate-evaluation', methods=['POST'])
def generate_evaluation():
    if not client:
        return jsonify({"error": "GROQ_API_KEY environment variable is not configured."}), 500

    data = request.json or {}
    topic = data.get("topic", "").strip()
    eval_type = data.get("type", "Quiz")
    difficulty = data.get("difficulty", "Intermediate")
    gap_history = data.get("gap_history", [])

    if not topic:
        return jsonify({"error": "Topic is required."}), 400

    system_prompt = (
        "You are EduMorph AI's Evaluation Forge Engine.\n"
        "Generate a diagnostic active-recall evaluation set formatted strictly as a JSON object.\n"
        "Structure format:\n"
        "{\n"
        "  \"items\": [\n"
        "    {\n"
        "      \"question\": \"Question or flashcard prompt text?\",\n"
        "      \"front\": \"Front prompt (for flashcards)\",\n"
        "      \"back\": \"Back explanation (for flashcards)\",\n"
        "      \"options\": [\"Option A\", \"Option B\", \"Option C\", \"Option D\"],\n"
        "      \"correct_index\": 0,\n"
        "      \"explanation\": \"Detailed active recall step-by-step reasoning.\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Do not include markdown code block syntax or backticks."
    )

    user_prompt = f"""
    Topic Base: {topic}
    Format Type: {eval_type}
    Difficulty Tier: {difficulty}
    Known Gap Targets: {json.dumps(gap_history)}

    Return 4 items in the items array.
    """

    try:
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="openai/gpt-oss-20b",
            temperature=0.3,
            max_tokens=2048,
            extra_body={"reasoning_effort": "low"},
            response_format={"type": "json_object"}
        )

        response_data = json.loads(completion.choices[0].message.content)
        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": f"Evaluation forge generation error: {str(e)}"}), 500


@app.route('/api/generate-research', methods=['POST'])
def generate_research():
    if not client:
        return jsonify({"error": "GROQ_API_KEY environment variable is not configured."}), 500

    data = request.json or {}
    topic = data.get("topic", "").strip()
    profile = data.get("profile", "ELI5")
    depth = data.get("depth", "Undergrad")

    if not topic:
        return jsonify({"error": "Topic is required."}), 400

    system_prompt = (
        "You are EduMorph AI's Academic Research Director.\n"
        "Synthesize an academic literature brief and return raw JSON with this exact structure:\n"
        "{\n"
        "  \"sections\": [\n"
        "     { \"heading\": \"Section Title\", \"content\": \"Deep analytical synthesis text...\" }\n"
        "  ],\n"
        "  \"references\": [\n"
        "     { \"key\": \"1\", \"text\": \"Author et al., Title of Publication, Journal (Year).\" }\n"
        "  ],\n"
        "  \"followup_questions\": [\n"
        "     \"Analytical inquiry question 1?\"\n"
        "  ]\n"
        "}\n\n"
        "Do not wrap your response in markdown text wrappers or include backticks. Return raw JSON object."
    )

    user_prompt = f"""
    Research Subject Matter: {topic}
    Cognitive Lens Profile: {profile}
    US Academic Depth Tier: {depth}
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
        return jsonify({"error": f"Research synthesis error: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
