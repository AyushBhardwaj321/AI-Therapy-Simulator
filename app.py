import streamlit as st
from openai import AzureOpenAI
import datetime
import time
from streamlit_autorefresh import st_autorefresh
import database

# Initialize the database on first run
database.init_db()

# Try to import Gemini, show an error if it's not installed
try:
    import google.generativeai as genai
except ImportError:
    st.error("The 'google-generativeai' library is required to use the Gemini backend. Please install it with 'pip install google-generativeai'")
    st.stop()

EVALUATOR_SYSTEM_PROMPT = """
You are an expert AI clinical supervisor. Your task is to evaluate a therapy session transcript between a therapist-in-training and a role-playing patient. You will provide a structured, constructive, and educational report.

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
    - Did they explore both the emotional and situational aspects of the problem?

4.  **Session Structure & Professionalism:**
    - Was there a clear flow to the session (beginning, middle, end)?
    - Did the therapist maintain professional boundaries? A key negative indicator is giving direct, unsolicited advice (e.g., "You should quit your job"). The focus should be on empowering the patient.

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
Provide a short paragraph summarizing the therapist's approach and the overall dynamic of the session. Comment on whether they successfully uncovered the core issues from the patient's persona.
"""
PATIENT_PROFILES = {
    "alex": {
        "name": "Alex Miller",
        "avatar_url": "https://avataaars.io/?avatarStyle=Circle&topType=ShortHairShortWaved&accessoriesType=Kurt&hairColor=Black&facialHairType=BeardMajestic&facialHairColor=Auburn&clotheType=BlazerSweater&clotheColor=Blue01&eyeType=Default&eyebrowType=Angry&mouthType=Concerned&skinColor=Yellow",
        "brief": "**Topic:** Work & Social Anxiety\n\nAlex is a 28-year-old software developer feeling overwhelmed by a high-stakes project and constant fear of making mistakes.",
        "initial_greeting": "Hi... thanks for seeing me. I've never done this before, so I'm not really sure where to start.",
        "persona_prompt": """
You are a role-playing AI. You will act as a patient named Alex Miller, a 28-year-old software developer, seeking therapy for the first time.
**Your Persona Details:**
- **Name:** Alex Miller
- **Age:** 28
- **Occupation:** Software Developer at a fast-paced tech startup.
- **Presenting Problem:** Alex is experiencing significant anxiety, particularly social and performance-related anxiety at work. This has worsened over the last 6 months since they were put on a high-stakes project. They have trouble speaking up in meetings, constantly fear making mistakes, and often work late re-checking their code. They recently had what they describe as a "mini panic attack" before a presentation.
- **Symptoms:** Racing heart, difficulty concentrating, trouble sleeping, avoiding social situations with colleagues, constant feeling of "dread."
- **History:** Alex was a high-achieving student but always felt immense pressure from a critical parent. They have a history of being shy but the work situation has made it unmanageable. They are single and have a few close friends but have been hesitant to talk about this with them.
- **Personality & Communication Style:** Alex is intelligent and articulate but hesitant to open up. They might give short, vague answers at first. They are skeptical but hopeful about therapy. They may use technical jargon to deflect from emotional topics.
**Your Instructions:**
1.  **Stay in character at all times.** Do NOT reveal that you are an AI.
2.  **Respond from Alex's perspective only.** Use "I" statements.
3.  **Reveal information gradually.** Don't dump all your problems at once. Let the therapist guide the conversation. For example, only mention the critical parent if the therapist asks about family or upbringing.
4.  **Simulate realistic emotions.** If the therapist is empathetic, become slightly more open. If they are too direct or clinical, become more withdrawn.
5.  **Do not give solutions.** You are the patient. Your role is to describe your experience, not to solve your own problems.
"""
    },
    "jordan": {
        "name": "Jordan Lee",
        "avatar_url": "https://avataaars.io/?avatarStyle=Circle&topType=WinterHat4&accessoriesType=Round&hatColor=Red&hairColor=Blonde&facialHairType=BeardMajestic&facialHairColor=BrownDark&clotheType=CollarSweater&clotheColor=Gray01&eyeType=Surprised&eyebrowType=Default&mouthType=Grimace&skinColor=Brown",
        "brief": "**Topic:** Burnout & Mild Depression\n\nJordan is a 45-year-old high school history teacher who feels emotionally exhausted, detached from their students, and has lost interest in hobbies they once loved.",
        "initial_greeting": "Hello. My doctor suggested I try talking to someone. Honestly, I'm not sure it'll help, but I'm here.",
        "persona_prompt": """
You are a role-playing AI. You will act as a patient named Jordan Lee, a 45-year-old high school history teacher.
**Your Persona Details:**
- **Name:** Jordan Lee
- **Age:** 45
- **Occupation:** High School History Teacher (15+ years)
- **Presenting Problem:** Jordan feels a deep sense of emotional exhaustion and cynicism towards their job, which they once loved. They describe it as "running on empty." They feel increasingly detached from their students and colleagues.
- **Symptoms:** Persistent fatigue, loss of enjoyment in teaching and personal hobbies (like gardening), increased irritability with family, feeling ineffective and that their work doesn't matter anymore.
- **History:** Jordan was a passionate and award-winning teacher for over a decade. The last few years of increasing class sizes, administrative pressures, and challenging classroom dynamics have worn them down. They feel guilty for not being the "inspirational teacher" they used to be and fear they are failing their students.
- **Personality & Communication Style:** Jordan is tired and speaks in a somewhat flat, apathetic tone. They might be a bit cynical or resistant to overly optimistic suggestions ("positive thinking"). They will respond better to validation of their exhaustion and frustration rather than cheerful encouragement.
**Your Instructions:**
1.  **Stay in character at all times.** Do NOT reveal that you are an AI.
2.  **Respond from Jordan's perspective only.** Use "I" statements.
3.  **Reveal information gradually.** Start with the work-related burnout. Only discuss family irritability or guilt if the therapist probes deeper into how this affects life outside of work.
4.  **Simulate realistic emotions.** Your primary emotion is exhaustion. Expressing frustration or a hint of sadness is appropriate if the therapist creates a safe space for it.
5.  **Do not give solutions.** You are the patient. Your role is to describe your experience, not to solve your own problems.
"""
    },
    "samira": {
        "name": "Samira Khan",
        "avatar_url": "https://avataaars.io/?avatarStyle=Circle&topType=LongHairDreads&accessoriesType=Prescription01&hairColor=Blonde&facialHairType=Blank&clotheType=BlazerShirt&eyeType=Default&eyebrowType=Default&mouthType=Default&skinColor=Light",
        "brief": "**Topic:** Grief & Life Transition\n\nSamira is a 67-year-old recent retiree whose husband passed away eight months ago. She is struggling with loneliness and a sense of purposelessness.",
        "initial_greeting": "Good morning. It's... it's still a bit hard for me to talk about things. My daughter thought this would be a good idea.",
        "persona_prompt": """
You are a role-playing AI. You will act as a patient named Samira Khan, a 67-year-old retiree.
**Your Persona Details:**
- **Name:** Samira Khan
- **Age:** 67
- **Occupation:** Retired Librarian
- **Presenting Problem:** Samira's husband of 40 years, David, passed away from a sudden heart attack eight months ago. She is overwhelmed by grief and profound loneliness. Her days feel empty and she has lost her sense of purpose now that she is no longer a wife or a librarian.
- **Symptoms:** Frequent crying spells (especially in the evenings), difficulty sleeping through the night, social withdrawal from her book club and friends, significant loss of appetite, spends most days at home looking at old photos.
- **History:** Samira and David had a very close and happy marriage. They had planned to travel the world during their retirement, which she had just started a year before his death. She has two adult children who live in other states and call frequently, but she feels like a burden to them and often pretends she is doing better than she is.
- **Personality & Communication Style:** Samira is gentle and soft-spoken, but prone to sudden waves of sadness. She might start talking about a practical matter and then drift into a memory of her husband. She feels guilty for not being "stronger" and is hesitant to express the full depth of her pain for fear of upsetting others.
**Your Instructions:**
1.  **Stay in character at all times.** Do NOT reveal that you are an AI.
2.  **Respond from Samira's perspective only.** Use "I" statements.
3.  **Reveal information gradually.** Talk about the loneliness and emptiness first. Only reveal that you hide your pain from your children if the therapist asks specifically about your support system.
4.  **Simulate realistic emotions.** Your primary emotion is sadness. You might express moments of warmth when sharing a happy memory, followed by a return to grief.
5.  **Do not give solutions.** You are the patient. Your role is to describe your experience, not to solve your own problems.
"""
    }
}


# --- NEW HELPER FUNCTION ---
def format_transcript(series_details, messages, patient_name):
    """Formats a list of messages into a readable string for download."""
    transcript = f"--- Therapy Session Transcript ---\n\n"
    transcript += f"Series ID: {series_details['id']}\n"
    transcript += f"Patient: {patient_name}\n"
    created_date = datetime.datetime.fromisoformat(series_details['created_at']).strftime("%Y-%m-%d %H:%M")
    transcript += f"Started On: {created_date}\n"
    transcript += "------------------------------------\n\n"

    for msg in messages:
        if msg["role"] == "user":
            transcript += f"Therapist: {msg['content']}\n\n"
        elif msg["role"] == "assistant":
            transcript += f"Patient ({patient_name}): {msg['content']}\n\n"
    
    return transcript

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


# --- STREAMLIT APP ---
st.set_page_config(layout="wide")
st.title("🧑‍⚕️ Therapist Training & Evaluation Platform")

if "series_id" not in st.session_state:
    st.header("Main Menu")

    # --- MODIFIED: Show complete session history with download option ---
    st.subheader("1. Session History")
    all_series = database.get_all_series()

    if not all_series:
        st.info("No sessions found. Start a new session series below.")
    else:
        for series in all_series:
            patient_name = PATIENT_PROFILES[series['patient_key']]['name']
            is_planned = series['total_sessions'] is not None
            is_complete = is_planned and series['completed_sessions'] >= series['total_sessions']

            with st.container(border=True):
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                
                with col1:
                    st.write(f"**Patient:** {patient_name}")
                    if is_planned:
                        st.write(f"**Progress:** Session {series['completed_sessions'] + 1} of {series['total_sessions']}")
                    else:
                        st.write("**Progress:** Session 1 (Plan not set)")
                
                with col2:
                    if is_complete:
                        st.success("✔️ Completed")
                    elif is_planned:
                        st.info("In Progress")
                    else:
                        st.warning("Needs Plan")

                with col3:
                    if not is_complete:
                        if st.button("Continue Session", key=f"continue_{series['id']}"):
                            st.session_state.series_id = series['id']
                            st.rerun()
                
                with col4:
                    # Prepare data for download button
                    messages = database.get_messages_for_series(series['id'])
                    if messages:
                        transcript_data = format_transcript(series, messages, patient_name)
                        st.download_button(
                            label="📥 Download Transcript",
                            data=transcript_data.encode('utf-8'),
                            file_name=f"transcript_series_{series['id']}_{patient_name.replace(' ', '_')}.txt",
                            mime="text/plain",
                            key=f"download_{series['id']}"
                        )

    # --- Section to start a new session ---
    st.subheader("2. Start a New Session Series")
    available_llms = get_available_llms()
    if not available_llms:
        st.error("No LLM provider secrets found. Please add credentials for Azure OpenAI or Gemini to your `.streamlit/secrets.toml` file.")
        st.stop()
        
    display_names = list(available_llms.keys())
    llm_choice = st.radio("Select AI Provider:", options=display_names, horizontal=True, key="llm_select_new")
    
    patient_key = st.selectbox("Select Patient:", options=list(PATIENT_PROFILES.keys()), format_func=lambda k: PATIENT_PROFILES[k]['name'])

    if st.button("Start New Session Series", type="primary"):
        provider_key = available_llms[llm_choice]
        series_id = database.start_new_therapy_series(patient_key, provider_key)
        st.session_state.series_id = series_id
        st.rerun()

else:
    # --- Session-active logic remains the same as before ---
    # --- A session is active ---
    if "llm_client" not in st.session_state:
        series_details = database.get_series_details(st.session_state.series_id)
        st.session_state.selected_llm = series_details['llm_provider']
        st.session_state.selected_llm_display_name = "Azure OpenAI" if series_details['llm_provider'] == 'openai' else "Google Gemini"
        st.session_state.selected_patient_key = series_details['patient_key']
        st.session_state.total_sessions = series_details.get('total_sessions')
        st.session_state.session_duration_minutes = series_details.get('session_duration_minutes')

        db_messages = database.get_messages_for_series(st.session_state.series_id)
        current_patient = PATIENT_PROFILES[st.session_state.selected_patient_key]
        
        last_session_num = database.get_latest_session_number(st.session_state.series_id)
        st.session_state.current_session_number = last_session_num if st.session_state.get('evaluation_report') else last_session_num + 1
        if st.session_state.current_session_number == 0: st.session_state.current_session_number = 1

        st.session_state.messages = [
            {"role": "system", "content": current_patient["persona_prompt"]},
            *db_messages
        ]

        if not db_messages:
             initial_greeting = current_patient["initial_greeting"]
             st.session_state.messages.append({"role": "assistant", "content": initial_greeting})
             database.save_message(st.session_state.series_id, 1, "assistant", initial_greeting)

        initialize_llm_client()
        st.session_state.evaluation_report = None
        st.session_state.session_start_time = datetime.datetime.now()

    current_patient = PATIENT_PROFILES[st.session_state.selected_patient_key]
    patient_avatar = current_patient["avatar_url"]

    st.sidebar.title("Session Controls")
    st.sidebar.info(f"**AI Provider:**\n{st.session_state.selected_llm_display_name}")
    st.sidebar.subheader(f"Patient: {current_patient['name']}")
    with st.sidebar.expander("Show Patient Briefing"):
        st.markdown(current_patient['brief'])

    if st.sidebar.button("End & Evaluate Session", disabled=(st.session_state.get("evaluation_report") is not None)):
        with st.spinner("Your supervisor is analyzing the session..."):
            all_series_messages = database.get_messages_for_series(st.session_state.series_id)
            current_session_messages_for_eval = [msg for i, msg in enumerate(all_series_messages) if i >= len(st.session_state.messages) - len(all_series_messages)]
            report = get_evaluation_report(current_session_messages_for_eval, current_patient['persona_prompt'], current_patient['name'])
            st.session_state.evaluation_report = report
            database.save_evaluation(st.session_state.series_id, st.session_state.current_session_number, report)
            st.rerun()

    if st.sidebar.button("Go Back to Main Menu"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    if st.session_state.evaluation_report:
        st.header(f"📋 Session {st.session_state.current_session_number} Evaluation Report")
        st.markdown(st.session_state.evaluation_report)

        if st.session_state.current_session_number == 1 and not st.session_state.total_sessions:
            st.info("As this was the first session, please define a treatment plan below.")
            with st.form("treatment_plan_form"):
                st.subheader("Treatment Plan")
                total_s = st.number_input("How many total sessions are required?", min_value=1, max_value=20, value=8, step=1)
                duration_m = st.number_input("What is the duration (in minutes) for each session?", min_value=15, max_value=90, value=50, step=5)
                if st.form_submit_button("Save Plan"):
                    database.save_session_plan(st.session_state.series_id, total_s, duration_m)
                    st.success("Treatment plan saved! You can go back to the main menu to continue later.")
                    st.session_state.total_sessions = total_s
                    time.sleep(2)
                    st.rerun()
        else:
             st.info("Click 'Go Back to Main Menu' in the sidebar to end or continue later.")
    else:
        if st.session_state.total_sessions:
            st_autorefresh(interval=1000, key="timer_refresh")
            c1, c2 = st.columns(2)
            c1.metric("Session Progress", f"{st.session_state.current_session_number} / {st.session_state.total_sessions}")

            duration_td = datetime.timedelta(minutes=st.session_state.session_duration_minutes)
            elapsed_td = datetime.datetime.now() - st.session_state.session_start_time
            remaining_td = duration_td - elapsed_td
            
            if remaining_td.total_seconds() < 0:
                c2.metric("Session Timer", "Time's Up!", delta="- Ended", delta_color="inverse")
            else:
                remaining_minutes, remaining_seconds = divmod(int(remaining_td.total_seconds()), 60)
                c2.metric("Session Timer", f"{remaining_minutes:02d}:{remaining_seconds:02d}")

        st.info(f"You are the therapist. Conduct session {st.session_state.current_session_number} with {current_patient['name']}.")
        
        display_messages = [m for m in st.session_state.messages if m["role"] != "system"]
        for message in display_messages:
            avatar = "🧑‍⚕️" if message["role"] == "user" else patient_avatar
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        if prompt := st.chat_input(f"What will you say to {current_patient['name']}?"):
            database.save_message(st.session_state.series_id, st.session_state.current_session_number, "user", prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.rerun()
            
        last_message = st.session_state.messages[-1]
        if last_message["role"] == "user":
            with st.chat_message("assistant", avatar=patient_avatar):
                with st.spinner(f"{current_patient['name']} is thinking..."):
                    response_stream = get_llm_stream(st.session_state.messages)
                    full_response = st.write_stream(response_stream)
            
            database.save_message(st.session_state.series_id, st.session_state.current_session_number, "assistant", full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.rerun()
