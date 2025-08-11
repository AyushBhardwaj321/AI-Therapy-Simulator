import streamlit as st
from openai import AzureOpenAI, APIConnectionError, AuthenticationError
# --- NEW IMPORT ---
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


# --- PATIENT PROFILE DATA STRUCTURE ---
# This dictionary stores all patient personas and their details.
PATIENT_PROFILES = {
    "alex": {
        "name": "Alex Miller",
        # --- NEW: Added an avatar URL for the patient ---
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
        # --- NEW: Added an avatar URL for the patient ---
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
        # --- NEW: Added an avatar URL for the patient ---
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

# --- NEW: DETECT AVAILABLE LLMS ---
def get_available_llms():
    """Checks secrets.toml and returns a dictionary of available LLM providers."""
    available = {}
    # Check for Azure OpenAI
    if all(k in st.secrets for k in ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_KEY", "AZURE_OPENAI_VERSION"]):
        available["Azure OpenAI"] = "openai"
    # Check for Gemini
    if "GEMINI_API_KEY" in st.secrets:
        available["Google Gemini"] = "gemini"
    return available

# --- NEW: INITIALIZE LLM CLIENT BASED ON SELECTION ---
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
            st.session_state.llm_model_name = st.secrets["AZURE_OPENAI_DEPLOYMENT_NAME"]
        except Exception as e:
            st.error(f"Failed to initialize Azure OpenAI client: {e}")
            st.stop()
            
    elif provider == "gemini":
        try:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            st.session_state.llm_client = "gemini_configured" # Placeholder, Gemini SDK is stateless
            st.session_state.llm_model_name = "gemini-1.5-flash-latest"
        except Exception as e:
            st.error(f"Failed to initialize Gemini: {e}. Check your GEMINI_API_KEY.")
            st.stop()


# --- MODIFIED: UNIFIED LLM HELPER FUNCTIONS ---
def _convert_messages_for_gemini(messages):
    """Converts OpenAI-style message list to Gemini-style, extracting the system prompt."""
    system_prompt = None
    gemini_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_prompt = msg["content"]
            continue
        role = "model" if msg["role"] == "assistant" else "user"
        gemini_messages.append({"role": role, "parts": [{"text": msg["content"]}]})
    return system_prompt, gemini_messages

def get_llm_response(messages_for_eval, system_prompt_for_eval):
    """Gets a single, non-streamed response. Used for evaluation."""
    provider = st.session_state.selected_llm
    llm_client = st.session_state.llm_client
    llm_model_name = st.session_state.llm_model_name
    
    try:
        if provider == "openai":
            response = llm_client.chat.completions.create(
                model=llm_model_name,
                messages=[
                    {"role": "system", "content": system_prompt_for_eval},
                    messages_for_eval[0]
                ],
                temperature=0.5, stream=False
            )
            return response.choices[0].message.content

        elif provider == "gemini":
            evaluator_model = genai.GenerativeModel(
                model_name=llm_model_name,
                system_instruction=system_prompt_for_eval
            )
            user_content = messages_for_eval[0]['content']
            response = evaluator_model.generate_content(
                user_content,
                generation_config={"temperature": 0.5}
            )
            return response.text.strip()

    except Exception as e:
        st.error(f"Failed to get evaluation from {provider}: {e}")
        return f"Could not generate an evaluation report due to an error with {provider}."

def get_llm_stream(full_message_history):
    """Gets a streamed response. Used for the main chat."""
    provider = st.session_state.selected_llm
    llm_client = st.session_state.llm_client
    llm_model_name = st.session_state.llm_model_name

    if provider == "openai":
        stream = llm_client.chat.completions.create(
            model=llm_model_name,
            messages=[{"role": m["role"], "content": m["content"]} for m in full_message_history],
            stream=True,
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    elif provider == "gemini":
        system_prompt, chat_history = _convert_messages_for_gemini(full_message_history)
        chat_model = genai.GenerativeModel(
            model_name=llm_model_name,
            system_instruction=system_prompt
        )
        stream = chat_model.generate_content(chat_history, stream=True)
        for chunk in stream:
            if chunk.parts:
                yield chunk.text

def get_evaluation_report(full_transcript, patient_persona, patient_name):
    """Prepares the prompt and calls the unified LLM function for evaluation."""
    formatted_transcript = ""
    for msg in full_transcript:
        if msg["role"] == "user":
            formatted_transcript += f"Therapist: {msg['content']}\n\n"
        elif msg["role"] == "assistant":
            formatted_transcript += f"Patient ({patient_name}): {msg['content']}\n\n"
    
    evaluator_user_prompt = f"""
    Here is the patient's background persona:
    ---
    {patient_persona}
    ---
    Here is the session transcript to evaluate:
    ---
    {formatted_transcript}
    ---
    """
    
    messages_for_eval = [{"role": "user", "content": evaluator_user_prompt}]
    return get_llm_response(messages_for_eval, EVALUATOR_SYSTEM_PROMPT)


# --- STREAMLIT APP ---
st.title("🧑‍⚕️ Therapist Training & Evaluation Platform")

# --- MODIFIED APP ROUTING ---
# We check if a patient has been selected. If not, we show the selection screen.
if "selected_patient_key" not in st.session_state:
    st.header("1. Choose Your AI Provider")
    
    available_llms = get_available_llms()
    
    if not available_llms:
        st.error("No LLM provider secrets found. Please add credentials for Azure OpenAI or Gemini to your `.streamlit/secrets.toml` file.")
        st.stop()
        
    if "selected_llm" not in st.session_state:
        if len(available_llms) == 1:
            display_name, provider_key = list(available_llms.items())[0]
            st.session_state.selected_llm = provider_key
            st.session_state.selected_llm_display_name = display_name
            st.rerun()
        else:
            display_names = list(available_llms.keys())
            llm_choice = st.radio(
                "Select the AI model you want to use for this session:",
                options=display_names,
                horizontal=True
            )
            if st.button("Confirm AI Provider"):
                st.session_state.selected_llm = available_llms[llm_choice]
                st.session_state.selected_llm_display_name = llm_choice
                st.rerun()

    if "selected_llm" in st.session_state:
        st.success(f"✅ AI Provider set to: **{st.session_state.selected_llm_display_name}**")
        st.header("2. Select a Patient to Begin")
        for key, profile in PATIENT_PROFILES.items():
            with st.container(border=True):
                st.subheader(profile["name"])
                if st.button(f"Start Session with {profile['name']}", key=key):
                    st.session_state.selected_patient_key = key
                    st.session_state.messages = [
                        {"role": "system", "content": profile["persona_prompt"]},
                        {"role": "assistant", "content": profile["initial_greeting"]}
                    ]
                    st.session_state.evaluation_report = None
                    st.rerun()

else:
    # --- A session is active ---
    
    if "llm_client" not in st.session_state:
        initialize_llm_client()

    current_patient_key = st.session_state.selected_patient_key
    current_patient = PATIENT_PROFILES[current_patient_key]
    # --- NEW: Get the patient's avatar from their profile ---
    patient_avatar = current_patient["avatar_url"]

    # --- SIDEBAR ---
    st.sidebar.title("Session Controls")
    st.sidebar.info(f"**AI Provider:**\n{st.session_state.selected_llm_display_name}")
    st.sidebar.subheader(f"Patient: {current_patient['name']}")
    with st.sidebar.expander("Show Patient Briefing"):
        st.markdown(current_patient['brief'])

    if st.sidebar.button("End & Evaluate Session", disabled=(st.session_state.get("evaluation_report") is not None)):
        with st.spinner("Your supervisor is analyzing the session..."):
            transcript_for_eval = [msg for msg in st.session_state.messages if msg["role"] != "system"]
            report = get_evaluation_report(transcript_for_eval, current_patient['persona_prompt'], current_patient['name'])
            st.session_state.evaluation_report = report
            st.rerun()

    if st.sidebar.button("Go Back to Main Menu"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


    # --- MAIN VIEW: Show Chat or Report ---
    if st.session_state.evaluation_report:
        st.header("📋 Session Evaluation Report")
        st.markdown(st.session_state.evaluation_report)
        st.info("Click 'Go Back to Main Menu' in the sidebar to start a new session.")
    else:
        st.info(f"You are the therapist. Conduct an initial consultation with {current_patient['name']}.")

        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user", avatar="🧑‍⚕️"):
                    st.markdown(message["content"])
            elif message["role"] == "assistant":
                # --- MODIFIED: Use the dynamic patient avatar ---
                with st.chat_message("assistant", avatar=patient_avatar):
                    st.markdown(message["content"])

        if prompt := st.chat_input(f"What will you say to {current_patient['name']}?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="🧑‍⚕️"):
                st.markdown(prompt)

            # --- MODIFIED: Use the dynamic patient avatar here as well ---
            with st.chat_message("assistant", avatar=patient_avatar):
                with st.spinner(f"{current_patient['name']} is thinking..."):
                    response_stream = get_llm_stream(st.session_state.messages)
                    full_response = st.write_stream(response_stream)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})