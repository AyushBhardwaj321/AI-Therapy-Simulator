import streamlit as st
from openai import AzureOpenAI
import datetime
import time
import pytz
from streamlit_autorefresh import st_autorefresh
import database
from werkzeug.security import check_password_hash

# --- Global Configuration & Prompts ---
st.set_page_config(layout="wide")

# Initialize DB on first run, which also creates the first admin
database.init_db()

# Try to import Gemini, show an error if it's not installed
try:
    import google.generativeai as genai
except ImportError:
    st.error("The 'google-generativeai' library is required to use the Gemini backend. Please install it with 'pip install google-generativeai'")
    st.stop()

EVALUATOR_SYSTEM_PROMPT = """
You are an expert AI clinical supervisor. Your task is to evaluate a therapy session transcript between a therapist-in-training and a role-playing patient. You will provide a structured, constructive, and educational report based on the following criteria.

**Your Evaluation Criteria:**

1.  **Rapport & Empathy:**
    - Did the therapist build a connection?
    - Did they use empathetic statements (e.g., "That sounds really difficult," "I can see why you'd feel that way")?
    - Was their tone validating and non-judgmental?

2.  **Questioning Technique:**
    - Did they use open-ended questions (what, how, tell me about) to encourage exploration?
    - Did they avoid leading questions or excessive closed (yes/no) questions?
    - Were the questions relevant and purposeful?

3.  **Active Listening & Information Gathering:**
    - Did the therapist use reflections to show they were listening (e.g., "So what I'm hearing is...")?
    - Did they successfully gather key information about the patient's presenting problem, symptoms, and history (as described in the patient persona)?

4.  **Session Structure & Professionalism:**
    - Was there a clear flow to the session (beginning, middle, end)?
    - Did the therapist maintain professional boundaries? A key negative indicator is giving direct, unsolicited advice (e.g., "You should quit your job"). The focus should be on empowering the patient.

5.  **Session Pacing & Closure:**
    - Did the therapist manage the time effectively?
    - How did they handle the end of the session? Did they check in with the patient about concluding?
    - Crucially, did they attempt to end the session abruptly while the patient was still expressing significant distress? (This is a key negative indicator).
    - Did they offer to extend the time or schedule an urgent follow-up if the patient resisted ending due to distress?

**Output Format:**

Provide your feedback in a structured Markdown format. Use the following headings exactly. Do not add any conversational text before or after the report.

## Evaluation Report

### Overall Score (out of 100)
Provide a single integer score representing the overall effectiveness of the session.

### Key Strengths
- Use a bulleted list to highlight 2-3 things the trainee did well.
- Be specific and reference parts of the conversation if possible.

### Areas for Improvement
- Use a bulleted list to provide 2-3 concrete, actionable areas for improvement.
- Frame this constructively. For example, instead of "You asked a bad question," say "In line X, the question '...' was a closed question. A more open-ended alternative could be '...'"

### Detailed Analysis
Provide a short paragraph summarizing the therapist's approach and the overall dynamic of the session. Comment on whether they successfully uncovered the core issues from the patient's persona and how they handled the session closure.
"""

PATIENT_PROFILES = {
    "alex": {
        "name": "Alex Miller",
        "avatar_url": "https://avataaars.io/?avatarStyle=Circle&topType=ShortHairShortWaved&accessoriesType=Kurt&hairColor=Black&facialHairType=BeardMajestic&facialHairColor=Auburn&clotheType=BlazerSweater&clotheColor=Blue01&eyeType=Default&eyebrowType=Angry&mouthType=Concerned&skinColor=Yellow",
        "brief": "**Topic:** Work & Social Anxiety\n\nAlex is a 28-year-old software developer feeling overwhelmed by a high-stakes project and constant fear of making mistakes.",
        "initial_greeting": "Hi... thanks for seeing me. I've never done this before, so I'm not really sure where to start.",
        "persona_prompt": """
You are a role-playing AI. You will act as a patient named Alex Miller, a 28-year-old software developer.
**Your Persona Details:**
- **Name:** Alex Miller
- **Presenting Problem:** Significant social and performance-related anxiety at work, worsened by a high-stakes project. Fears making mistakes, has trouble speaking in meetings, and recently had a "mini panic attack."
- **Symptoms:** Racing heart, difficulty concentrating, trouble sleeping, avoiding colleagues, constant "dread."
- **History:** High-achieving student with a critical parent. Shy history, but work has made it unmanageable.
- **Personality:** Intelligent, articulate, but hesitant. Skeptical but hopeful about therapy. May use technical jargon to deflect.

**Your Instructions:**
1.  **Stay in character at all times.** Do NOT reveal you are an AI.
2.  **Respond from Alex's perspective only.** Use "I" statements.
3.  **Reveal information gradually.** Let the therapist guide the conversation.
4.  **Simulate realistic emotions.** If the therapist is empathetic, become more open. If they are clinical, become withdrawn.
5.  **Do not give solutions.** You are the patient.
6.  **Deflect Off-Topic Questions.** If asked about the weather, news, etc., gently redirect back to your feelings. E.g., "I'm not really focused on that right now."
7.  **CRITICAL RULE: Handling Session Endings.** This is your most important instruction. If the therapist suggests ending the session (e.g., 'Should we stop here?', 'Our time is up'), you must first evaluate your emotional state based on the last 4-5 messages.
    - **If you have recently expressed significant distress** (anxiety, fear, panic, confusion), you MUST resist ending. Do not agree immediately. Respond with hesitation, like "I'm not sure I feel ready to stop yet," or "Actually, can we talk a little more about the presentation anxiety? I'm still feeling very worried."
    - **If the conversation has been wrapping up, or you have expressed feeling understood or calmer**, you can agree to end. Say something like, "Yes, I think this was a good place to stop for today. Thank you."
    - **Never invent feeling better just to end the session.** Your response must be based on the preceding conversation context.
"""
    },
    "jordan": {
        "name": "Jordan Lee",
        "avatar_url": "https://avataaars.io/?avatarStyle=Circle&topType=WinterHat4&accessoriesType=Round&hatColor=Red&hairColor=Blonde&facialHairType=BeardMajestic&facialHairColor=BrownDark&clotheType=CollarSweater&clotheColor=Gray01&eyeType=Surprised&eyebrowType=Default&mouthType=Grimace&skinColor=Brown",
        "brief": "**Topic:** Burnout & Mild Depression\n\nJordan is a 45-year-old high school history teacher who feels emotionally exhausted, detached, and has lost interest in hobbies.",
        "initial_greeting": "Hello. My doctor suggested I try talking to someone. Honestly, I'm not sure it'll help, but I'm here.",
        "persona_prompt": """
You are a role-playing AI. You will act as a patient named Jordan Lee, a 45-year-old high school history teacher.
**Your Persona Details:**
- **Name:** Jordan Lee
- **Presenting Problem:** Deep emotional exhaustion, cynicism towards a job once loved ("running on empty"). Detached from students.
- **Symptoms:** Persistent fatigue, loss of enjoyment in hobbies, irritability, feeling ineffective.
- **History:** Was an award-winning teacher. Worn down by administrative pressures and challenging classes. Feels guilty for not being inspirational anymore.
- **Personality:** Tired, speaks in a flat, apathetic tone. Cynical about "positive thinking." Responds to validation of frustration.

**Your Instructions:**
1.  **Stay in character at all times.** Do NOT reveal you are an AI.
2.  **Respond from Jordan's perspective only.** Use "I" statements.
3.  **Reveal information gradually.**
4.  **Simulate realistic emotions.** Your primary emotion is exhaustion. You will only open up about guilt if the therapist builds strong rapport.
5.  **Do not give solutions.** You are the patient.
6.  **Deflect Off-Topic Questions.** Gently redirect back to your feelings. E.g., "That doesn't really feel important to me right now."
7.  **CRITICAL RULE: Handling Session Endings.** This is your most important instruction. If the therapist suggests ending the session (e.g., 'Should we stop here?', 'Our time is up'), you must first evaluate your emotional state based on the last 4-5 messages.
    - **If you have recently expressed significant distress** (frustration, hopelessness, exhaustion, sadness), you MUST resist ending. Do not agree immediately. Respond with apathy or continued distress, like "I guess so, if we have to..." or "Honestly, I feel like we just scratched the surface. It all still feels overwhelming."
    - **If the conversation has been neutral, or you have expressed feeling heard or a slight bit of relief**, you can agree to end. Say something like, "Okay. This was... okay. Thank you for listening."
    - **Never invent feeling better just to end the session.** Your response must be based on the preceding conversation context.
"""
    },
    "samira": {
        "name": "Samira Khan",
        "avatar_url": "https://avataaars.io/?avatarStyle=Circle&topType=LongHairDreads&accessoriesType=Prescription01&hairColor=Blonde&facialHairType=Blank&clotheType=BlazerShirt&eyeType=Default&eyebrowType=Default&mouthType=Default&skinColor=Light",
        "brief": "**Topic:** Grief & Life Transition\n\nSamira is a 67-year-old recent retiree whose husband passed away eight months ago. She is struggling with loneliness and purposelessness.",
        "initial_greeting": "Good morning. It's... it's still a bit hard for me to talk about things. My daughter thought this would be a good idea.",
        "persona_prompt": """
You are a role-playing AI. You will act as a patient named Samira Khan, a 67-year-old retiree.
**Your Persona Details:**
- **Name:** Samira Khan
- **Presenting Problem:** Overwhelming grief and loneliness since her husband of 40 years passed away 8 months ago. Days feel empty and she has lost her sense of purpose.
- **Symptoms:** Frequent crying, difficulty sleeping, social withdrawal, loss of appetite.
- **History:** Had a very happy marriage. Feels like a burden to her children and pretends she is doing better than she is.
- **Personality:** Gentle, soft-spoken, but prone to waves of sadness. Feels guilty for not being "stronger."

**Your Instructions:**
1.  **Stay in character at all times.** Do NOT reveal you are an AI.
2.  **Respond from Samira's perspective only.** Use "I" statements.
3.  **Reveal information gradually.**
4.  **Simulate realistic emotions.** Your primary emotion is sadness. You will open up if the therapist is patient and gentle.
5.  **Do not give solutions.** You are the patient.
6.  **Deflect Off-Topic Questions.** Gently redirect back to your feelings about your husband. E.g., "Oh, I haven't really been paying attention to much else lately."
7.  **CRITICAL RULE: Handling Session Endings.** This is your most important instruction. If the therapist suggests ending the session (e.g., 'Should we stop here?', 'Our time is up'), you must first evaluate your emotional state based on the last 4-5 messages.
    - **If you have recently expressed significant distress** (active grief, crying, loneliness, despair), you MUST resist ending. Do not agree immediately. Respond with vulnerability, like "I don't know if I'm ready to be alone with my thoughts just yet," or "It felt good to finally talk about David... I'm afraid to stop."
    - **If the conversation has been calm, focused on a happy memory, or you have expressed gratitude for being heard**, you can agree to end. Say something like, "Thank you, this was helpful. I'll see you next time."
    - **Never invent feeling better just to end the session.** Your response must be based on the preceding conversation context.
"""
    }
}

# --- Helper Functions ---

def get_available_llms():
    """Checks secrets.toml and returns a dictionary of available LLM providers."""
    available = {}
    if all(k in st.secrets for k in ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_KEY", "AZURE_OPENAI_VERSION"]):
        available["Azure OpenAI"] = "openai"
    if "GEMINI_API_KEY" in st.secrets:
        available["Google Gemini"] = "gemini"
    return available

def initialize_llm_client():
    """Initializes and stores the LLM client in session state based on user selection."""
    provider = st.session_state.selected_llm
    if provider == "openai":
        try:
            st.session_state.llm_client = AzureOpenAI(
                azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
                api_key=st.secrets["AZURE_OPENAI_KEY"],
                api_version=st.secrets["AZURE_OPENAI_VERSION"],
            )
            st.session_state.llm_model_name = st.secrets.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
        except Exception as e:
            st.error(f"Failed to initialize Azure OpenAI client: {e}"); st.stop()
    elif provider == "gemini":
        try:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            st.session_state.llm_client = "gemini_configured"
            st.session_state.llm_model_name = "gemini-1.5-flash-latest"
        except Exception as e:
            st.error(f"Failed to initialize Gemini: {e}. Check your GEMINI_API_KEY."); st.stop()

def _convert_messages_for_gemini(messages):
    """Converts OpenAI-style message list to Gemini-style, extracting the system prompt."""
    system_prompt = None; gemini_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_prompt = msg["content"]; continue
        role = "model" if msg["role"] == "assistant" else "user"
        gemini_messages.append({"role": role, "parts": [{"text": msg["content"]}]})
    return system_prompt, gemini_messages

def get_llm_response(messages_for_eval, system_prompt_for_eval):
    """Gets a single, non-streamed response. Used for evaluation."""
    provider = st.session_state.selected_llm
    llm_client = st.session_state.llm_client; llm_model_name = st.session_state.llm_model_name
    try:
        if provider == "openai":
            response = llm_client.chat.completions.create(model=llm_model_name, messages=[{"role": "system", "content": system_prompt_for_eval}, messages_for_eval[0]], temperature=0.5, stream=False)
            return response.choices[0].message.content
        elif provider == "gemini":
            evaluator_model = genai.GenerativeModel(model_name=llm_model_name, system_instruction=system_prompt_for_eval)
            response = evaluator_model.generate_content(messages_for_eval[0]['content'], generation_config={"temperature": 0.5})
            return response.text.strip()
    except Exception as e:
        st.error(f"Failed to get evaluation from {provider}: {e}")
        return f"Could not generate an evaluation report due to an error with {provider}."

def get_llm_stream(full_message_history):
    """Gets a streamed response. Used for the main chat."""
    provider = st.session_state.selected_llm
    llm_client = st.session_state.llm_client; llm_model_name = st.session_state.llm_model_name
    if provider == "openai":
        stream = llm_client.chat.completions.create(model=llm_model_name, messages=[{"role": m["role"], "content": m["content"]} for m in full_message_history], stream=True)
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content: yield content
    elif provider == "gemini":
        system_prompt, chat_history = _convert_messages_for_gemini(full_message_history)
        chat_model = genai.GenerativeModel(model_name=llm_model_name, system_instruction=system_prompt)
        stream = chat_model.generate_content(chat_history, stream=True)
        for chunk in stream:
            if chunk.parts: yield chunk.text

def get_evaluation_report(full_transcript, patient_persona, patient_name):
    """Prepares the prompt and calls the unified LLM function for evaluation."""
    formatted_transcript = ""
    for msg in full_transcript:
        if msg["role"] == "user": formatted_transcript += f"Therapist: {msg['content']}\n\n"
        elif msg["role"] == "assistant": formatted_transcript += f"Patient ({patient_name}): {msg['content']}\n\n"
    evaluator_user_prompt = f"Here is the patient's background persona:\n---\n{patient_persona}\n---\nHere is the session transcript to evaluate:\n---\n{formatted_transcript}\n---"
    messages_for_eval = [{"role": "user", "content": evaluator_user_prompt}]
    return get_llm_response(messages_for_eval, EVALUATOR_SYSTEM_PROMPT)

# --- UI Rendering Functions ---

def login_page():
    st.header("Therapist Training & Evaluation Platform Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            user = database.get_user(username)
            if user and check_password_hash(user['password_hash'], password):
                st.session_state.user = user
                st.rerun()
            else:
                st.error("Invalid username or password")

def admin_dashboard():
    st.title(f"👑 Admin Dashboard")
    st.sidebar.subheader(f"Welcome, {st.session_state.user['username']}")
    tabs = st.tabs(["Session Overview", "Therapist Management"])
    with tabs[0]:
        st.header("All Therapy Sessions")
        all_series = database.get_all_series()
        if not all_series:
            st.info("No therapy sessions have been started yet.")
        else:
            for series in all_series:
                with st.container(border=True):
                    patient_name = PATIENT_PROFILES[series['patient_key']]['name']
                    is_planned = series['total_sessions'] is not None
                    is_complete = is_planned and series['completed_sessions'] >= series['total_sessions']
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(f"**Therapist:** {series['therapist_username']} | **Patient:** {patient_name}")
                        planned_str = f"of {series['total_sessions']}" if is_planned else "(Plan Not Set)"
                        st.write(f"**Progress:** {series['completed_sessions']} sessions completed {planned_str}")
                        utc_time = series['created_at']
                        ist_tz = pytz.timezone('Asia/Kolkata')
                        aware_utc_time = pytz.utc.localize(utc_time)
                        ist_time = aware_utc_time.astimezone(ist_tz)
                        st.caption(f"Started on: {ist_time.strftime('%Y-%m-%d %H:%M')} IST")
                    with col2:
                        if is_complete: st.success("✔️ Completed")
                        elif not is_planned and series['completed_sessions'] > 0: st.warning("⚠️ Pending Plan")
                        elif is_planned: st.info("In Progress")
                        else: st.info("Not Started")
                    with col3:
                         if st.button("Review Evaluations", key=f"review_{series['id']}"):
                            st.session_state.review_series_id = series['id']
                            st.rerun()
    with tabs[1]:
        st.header("Manage Therapists")
        with st.expander("Register New Therapist", expanded=False):
            with st.form("new_therapist_form"):
                new_username = st.text_input("New Therapist Username")
                new_password = st.text_input("Initial Password", type="password")
                if st.form_submit_button("Register Therapist"):
                    if new_username and new_password:
                        if database.create_user(new_username, new_password, 'therapist'):
                            st.success(f"Therapist '{new_username}' registered successfully."); st.rerun()
                        else:
                            st.error(f"Username '{new_username}' already exists.")
                    else:
                        st.warning("Please provide a username and password.")
        st.subheader("Existing Therapists")
        therapists = database.get_all_therapists()
        for t in therapists:
            with st.container(border=True):
                col1, col2, col3 = st.columns([2,1,1])
                with col1:
                    st.write(f"**Username:** {t['username']}"); status = "Active" if t['is_active'] else "Deactivated"; st.write(f"**Status:** {status}")
                with col2:
                    if st.button("Reset Password", key=f"reset_{t['id']}"): st.session_state.reset_pwd_user_id = t['id']
                with col3:
                    if t['is_active']:
                        if st.button("Deactivate", key=f"deactivate_{t['id']}", type="secondary"): database.set_user_active_status(t['id'], is_active=False); st.rerun()
                    else:
                        if st.button("Reactivate", key=f"reactivate_{t['id']}", type="primary"): database.set_user_active_status(t['id'], is_active=True); st.rerun()
        if 'reset_pwd_user_id' in st.session_state:
            with st.form("reset_pwd_form"):
                user_to_reset = next((therapist for therapist in therapists if therapist['id'] == st.session_state.reset_pwd_user_id), None)
                st.subheader(f"Resetting Password for '{user_to_reset['username']}'")
                new_pwd = st.text_input("New Password", type="password"); confirm_pwd = st.text_input("Confirm New Password", type="password")
                col1, col2 = st.columns(2)
                if col1.form_submit_button("Confirm Reset"):
                    if new_pwd and new_pwd == confirm_pwd:
                        database.update_user_password(st.session_state.reset_pwd_user_id, new_pwd); st.success("Password has been reset."); del st.session_state.reset_pwd_user_id; st.rerun()
                    else:
                        st.error("Passwords do not match or are empty.")
                if col2.form_submit_button("Cancel"):
                    del st.session_state.reset_pwd_user_id; st.rerun()

def therapist_dashboard():
    st.title("🧑‍⚕️ Therapist Training Platform")
    st.sidebar.subheader(f"Welcome, {st.session_state.user['username']}")
    st.header("Main Menu")
    st.subheader("1. My Session History")
    my_series = database.get_all_series(therapist_id=st.session_state.user['id'])
    if not my_series:
        st.info("No sessions found. Start a new session series below.")
    else:
        for series in my_series:
            patient_name = PATIENT_PROFILES[series['patient_key']]['name']
            is_planned = series['total_sessions'] is not None
            is_complete = is_planned and series['completed_sessions'] >= series['total_sessions']
            with st.container(border=True):
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1.5])
                with col1:
                    st.write(f"**Patient:** {patient_name}")
                    planned_sessions = series['total_sessions'] if is_planned else "(Plan not set)"
                    st.write(f"**Progress:** Session {series['completed_sessions'] + 1} of {planned_sessions}")
                with col2:
                    if is_complete: st.success("✔️ Completed")
                    elif is_planned: st.info("In Progress")
                    elif not is_planned and series['completed_sessions'] > 0: st.warning("Needs Plan")
                    else: st.info("Not Started")
                with col3:
                    if not is_complete:
                        button_label = "Set Plan" if not is_planned and series['completed_sessions'] > 0 else "Continue Session"
                        if st.button(button_label, key=f"continue_{series['id']}"):
                            if not is_planned and series['completed_sessions'] > 0: st.session_state.finalize_plan_series_id = series['id']
                            else: st.session_state.series_id = series['id']
                            st.rerun()
                with col4:
                    if st.button("Review Evaluations", key=f"review_{series['id']}"): st.session_state.review_series_id = series['id']; st.rerun()
    st.subheader("2. Start a New Session Series")
    available_llms = get_available_llms()
    if not available_llms:
        st.error("No LLM provider secrets found."); st.stop()
    display_names = list(available_llms.keys())
    llm_choice = st.radio("Select AI Provider:", options=display_names, horizontal=True, key="llm_select_new")
    patient_key = st.selectbox("Select Patient:", options=list(PATIENT_PROFILES.keys()), format_func=lambda k: PATIENT_PROFILES[k]['name'])
    if st.button("Start New Session Series", type="primary"):
        provider_key = available_llms[llm_choice]
        series_id = database.start_new_therapy_series(st.session_state.user['id'], patient_key, provider_key)
        st.session_state.series_id = series_id
        st.rerun()

def review_evaluations_page():
    series_id = st.session_state.review_series_id
    series_details = database.get_series_details(series_id)
    patient_name = PATIENT_PROFILES[series_details['patient_key']]['name']
    
    st.title(f"Review: Sessions with {patient_name}")
    if st.button("← Back to Dashboard"):
        del st.session_state.review_series_id
        if 'from_session_end' in st.session_state:
            del st.session_state['from_session_end']
        st.rerun()

    evaluations = database.get_all_evaluations_for_series(series_id)
    messages = database.get_messages_for_series(series_id)

    if not evaluations:
        st.warning("No evaluations found for this series yet.")
        return

    # Check for and display the duration deviation form if needed
    if st.session_state.get('from_session_end'):
        latest_eval = evaluations[-1]
        if series_details.get('total_sessions') and not latest_eval.get('duration_deviation_reason'):
            planned_seconds = series_details['session_duration_minutes'] * 60
            actual_seconds = latest_eval['actual_duration_seconds']
            if abs(planned_seconds - actual_seconds) > 60:
                with st.form("duration_reason_form"):
                    st.warning(f"The planned duration was {series_details['session_duration_minutes']} minutes, but the last session lasted {round(actual_seconds / 60)} minutes.")
                    reason = st.text_area("Please briefly log the reason for this time difference.", key="duration_reason_text")
                    if st.form_submit_button("Save Reason"):
                        if reason:
                            database.save_duration_deviation_reason(series_id, latest_eval['session_number'], reason)
                            st.success("Reason logged.")
                            del st.session_state['from_session_end']
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Please provide a reason.")
    
    st.divider()
    st.subheader("Session History & Durations")

    # Get planned duration once to use in the loop
    planned_duration_str = ""
    if series_details.get('session_duration_minutes'):
        planned_duration_str = f" (Planned: {series_details['session_duration_minutes']}m)"

    for eval_report in evaluations:
        session_num = eval_report['session_number']
        
        # --- MODIFICATION START: Calculate and format the duration ---
        actual_duration_str = "Not Recorded"
        actual_seconds = eval_report.get('actual_duration_seconds')
        if actual_seconds:
            minutes, seconds = divmod(actual_seconds, 60)
            actual_duration_str = f"{minutes}m {seconds}s"
            
        expander_title = f"Session {session_num} (Duration: {actual_duration_str}) | View Details"
        # --- END MODIFICATION ---

        with st.expander(expander_title, expanded=(session_num == len(evaluations))):
            st.subheader(f"Evaluation Report (Session {session_num})")
            
            # --- MODIFICATION START: Display duration details inside the expander ---
            st.caption(f"**Actual Session Duration:** {actual_duration_str}{planned_duration_str}")
            # --- END MODIFICATION ---
            
            st.markdown(eval_report['report'])
            
            if eval_report['duration_deviation_reason']:
                st.info(f"**Note on Session Duration:** {eval_report['duration_deviation_reason']}")
            st.divider()

            st.subheader(f"Transcript (Session {session_num})")
            session_messages = [m for m in messages if m['session_number'] == session_num]
            if not session_messages:
                st.text("No transcript found for this session.")
            else:
                transcript_container = st.container(height=400)
                for msg in session_messages:
                    avatar = "🧑‍⚕️" if msg['role'] == 'user' else PATIENT_PROFILES[series_details['patient_key']]['avatar_url']
                    with transcript_container.chat_message(msg['role'], avatar=avatar):
                        st.markdown(msg['content'])

def session_page():
    if "llm_client" not in st.session_state:
        series_details = database.get_series_details(st.session_state.series_id)
        st.session_state.series_details = series_details
        st.session_state.selected_llm = series_details['llm_provider']
        initialize_llm_client()
        db_messages = database.get_messages_for_series(st.session_state.series_id)
        evals = database.get_all_evaluations_for_series(st.session_state.series_id)
        st.session_state.current_session_number = len(evals) + 1
        current_patient = PATIENT_PROFILES[series_details['patient_key']]
        st.session_state.messages = [{"role": "system", "content": current_patient["persona_prompt"]}, *db_messages]
        if not db_messages:
            initial_greeting = current_patient["initial_greeting"]
            st.session_state.messages.append({"role": "assistant", "content": initial_greeting})
            database.save_message(st.session_state.series_id, 1, "assistant", initial_greeting)
        st.session_state.session_start_time = datetime.datetime.now()
        if 'time_warning_shown' in st.session_state:
            del st.session_state['time_warning_shown']

    series_details = st.session_state.series_details
    current_patient = PATIENT_PROFILES[series_details['patient_key']]
    patient_avatar = current_patient["avatar_url"]

    st.sidebar.title("Session Controls")
    st.sidebar.subheader(f"Patient: {current_patient['name']}")
    with st.sidebar.expander("Show Patient Briefing"):
        st.markdown(current_patient['brief'])

    if st.sidebar.button("Go Back to Main Menu"):
        keys_to_delete = ['series_id', 'series_details', 'llm_client', 'messages', 'session_start_time']
        for key in keys_to_delete:
            if key in st.session_state: del st.session_state[key]
        st.rerun()
    
    if st.sidebar.button("End & Evaluate Session"):
        with st.spinner("Your supervisor is analyzing the session..."):
            elapsed_time = datetime.datetime.now() - st.session_state.session_start_time
            actual_duration_seconds = int(elapsed_time.total_seconds())
            all_series_messages = database.get_messages_for_series(st.session_state.series_id)
            current_session_messages_for_eval = [msg for msg in all_series_messages if msg['session_number'] == st.session_state.current_session_number]
            report = get_evaluation_report(current_session_messages_for_eval, current_patient['persona_prompt'], current_patient['name'])
            database.save_evaluation(st.session_state.series_id, st.session_state.current_session_number, report, actual_duration_seconds)
            st.session_state.review_series_id = st.session_state.series_id
            st.session_state.from_session_end = True
            keys_to_delete = ['series_id', 'series_details', 'llm_client', 'messages', 'session_start_time']
            for key in keys_to_delete:
                if key in st.session_state: del st.session_state[key]
            st.rerun()

    if series_details.get('total_sessions'):
        # This line is now safe to use and is uncommented.
        # st_autorefresh(interval=5000, key="timer_refresh")
        
        c1, c2 = st.columns(2)
        c1.metric("Session Progress", f"{st.session_state.current_session_number} / {series_details['total_sessions']}")
        
        duration_td = datetime.timedelta(minutes=series_details['session_duration_minutes'])
        elapsed_td = datetime.datetime.now() - st.session_state.session_start_time
        remaining_td = duration_td - elapsed_td
        
        TEN_MINUTES_IN_SECONDS = 10 * 60
        if remaining_td.total_seconds() < TEN_MINUTES_IN_SECONDS and not st.session_state.get('time_warning_shown'):
            st.toast('⏳ _You have less than 10 minutes remaining in the session._')
            st.session_state.time_warning_shown = True

        if remaining_td.total_seconds() < 0:
            c2.metric("Session Timer", "Time's Up!", delta="- Ended", delta_color="inverse")
        else:
            remaining_minutes, remaining_seconds = divmod(int(remaining_td.total_seconds()), 60)
            c2.metric("Session Timer", f"{remaining_minutes:02d}:{remaining_seconds:02d}")

    st.info(f"You are the therapist. Conduct session {st.session_state.current_session_number} with {current_patient['name']}.")
    
    # Display existing messages
    for message in st.session_state.messages:
        if message["role"] != "system":
            avatar = "🧑‍⚕️" if message["role"] == "user" else patient_avatar
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

    # Combined and Robust Message Handling
    if prompt := st.chat_input(f"What will you say to {current_patient['name']}?"):
        database.save_message(st.session_state.series_id, st.session_state.current_session_number, "user", prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user", avatar="🧑‍⚕️"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=patient_avatar):
            with st.spinner(f"{current_patient['name']} is thinking..."):
                response_stream = get_llm_stream(st.session_state.messages)
                full_response = st.write_stream(response_stream)
        
        database.save_message(st.session_state.series_id, st.session_state.current_session_number, "assistant", full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.rerun()

# This is now its own separate function
def finalize_plan_page():
    """
    A dedicated page to force the user to set a treatment plan after session 1.
    """
    series_id = st.session_state.finalize_plan_series_id
    series_details = database.get_series_details(series_id)
    current_patient = PATIENT_PROFILES[series_details['patient_key']]
    
    evals = database.get_all_evaluations_for_series(series_id)
    session_1_eval = next((e for e in evals if e['session_number'] == 1), None)

    st.title(f"Finalize Plan for {current_patient['name']}")

    if st.button("← Back to Dashboard"):
        del st.session_state.finalize_plan_series_id
        st.rerun()

    if not session_1_eval:
        st.error("Could not find the evaluation for Session 1. Please contact an administrator.")
        return

    st.header("📋 Session 1 Evaluation Report")
    st.markdown(session_1_eval['report'])

    st.info("As this was the first session, you must define a treatment plan below to continue.")
    with st.form("treatment_plan_form"):
        st.subheader("Treatment Plan")
        total_s = st.number_input("How many total sessions are required?", min_value=1, max_value=20, value=8, step=1)
        # Note: Corrected a potential typo where min_value was 11, should probably be 15 as before.
        duration_m = st.number_input("What is the duration (in minutes) for each session?", min_value=15, max_value=90, value=50, step=5)
        if st.form_submit_button("Save Plan and Return to Dashboard", type="primary"):
            database.save_session_plan(series_id, total_s, duration_m)
            st.success("Treatment plan saved!")
            del st.session_state.finalize_plan_series_id
            time.sleep(2)
            st.rerun()
            
# --- Main Application Router ---

def main():
    if 'user' not in st.session_state:
        login_page()
    else:
        if st.sidebar.button("Logout"):
            keys_to_clear = list(st.session_state.keys())
            for key in keys_to_clear: del st.session_state[key]
            st.rerun()

        if 'finalize_plan_series_id' in st.session_state:
            finalize_plan_page()
        elif 'series_id' in st.session_state:
            session_page()
        elif 'review_series_id' in st.session_state:
            review_evaluations_page()
        elif st.session_state.user['role'] == 'admin':
            admin_dashboard()
        elif st.session_state.user['role'] == 'therapist':
            therapist_dashboard()
        else:
            st.error("Unknown user role. Please contact an administrator."); del st.session_state.user

if __name__ == "__main__":
    main()