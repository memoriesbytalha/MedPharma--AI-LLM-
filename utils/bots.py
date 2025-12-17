from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import streamlit as st
import warnings
from dotenv import load_dotenv
import os,sys
sys.path.append(os.getcwd())
load_dotenv()  # Load .env variables

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
warnings.warn("deprecated", DeprecationWarning)

OPENROUTER_API_KEY = os.getenv('OPEN_API_ROUTER_KEY')
def create_explanation_bot(openrouter_api_key):
    """Initialize LangChain bot for drug interaction explanations using OpenRouter"""
    
    llm = ChatOpenAI(
        model="arcee-ai/trinity-mini:free",  # Free model on OpenRouter
        openai_api_key=openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.7,
        max_tokens=2000
    )

    # Define the System (AI Persona) Message
    system_template = """
    You are a medical and chemical assistant AI helping explain drug interactions to non-specialists.
    Avoid complex jargon. Use clear, layman-friendly language.
    """
    system_message = SystemMessagePromptTemplate.from_template(system_template)

    # Define the Human (User Input/Data) Message
    human_template = """
        Predicted interaction between:
        - Drug A: {drug_a} (Formula: {drug_a_formula})
        - Drug B: {drug_b} (Formula: {drug_b_formula})
        - Severity: {prediction}
        - Confidence: {confidence}

        Important pathways:
        {pathways}

        Explain in **clear layman-friendly language**:
        1. Clinical meaning
        2. How chemical composition might contribute
        3. Precautions or concerns
        """
    human_message = HumanMessagePromptTemplate.from_template(human_template)
    
    # Create the ChatPromptTemplate using the messages list
    prompt_template = ChatPromptTemplate.from_messages([
        system_message,
        human_message
    ])
    # The chain logic was incorrect in the original code, 
    # as llm() is not the correct way to initialize the runnable
    # Correct LangChain V2 (Runnable) approach:
    layman_chain = prompt_template | llm | StrOutputParser() # Assuming StrOutputParser is used later
    
    return layman_chain
def pharmacologist_chatbot():
    """Chatbot interface for pharmacologists to interact with LLM"""
    
    st.title("💬 Pharmacologist Assistant Chatbot")
    st.markdown("Ask questions about drug interactions, pharmacology, molecular biology, and more.")
    
    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # System message for the LLM
    system_prompt = """You are an expert pharmacologist and medicinal chemist assistant helping healthcare professionals understand drug interactions, pharmacokinetics, pharmacodynamics, and molecular biology.

Provide accurate, evidence-based information in professional but accessible language. When discussing interactions:
- Explain mechanisms clearly
- Reference relevant pathways (CYP450, transporters, etc.)
- Mention severity and clinical significance
- Suggest monitoring parameters when relevant
- Always recommend consulting clinical references for critical decisions

Be concise but thorough."""
    
    # Display chat history
    st.markdown("### Conversation History")
    chat_container = st.container(border=True, height=400)
    
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div style='background-color: rgba(0, 0, 0, 0.6); padding: 12px; border-radius: 8px; margin: 8px 0;'>
                    <b>👤 You:</b><br>{message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background-color: rgba(0, 0, 0, 0.6); padding: 12px; border-radius: 8px; margin: 8px 0;'>
                    <b>🤖 Assistant:</b><br>{message["content"]}
                </div>
                """, unsafe_allow_html=True)
    
    # Input section
    st.markdown("---")
    st.markdown("### Ask a Question")
    
    user_input = st.text_area(
        "Your question:",
        placeholder="e.g., What are the major CYP450 interactions of warfarin? Explain the mechanism of this drug interaction...",
        height=80
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        send_button = st.button("📤 Send", type="primary", use_container_width=True)
    
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    with col3:
        st.markdown("<div></div>", unsafe_allow_html=True)
    
    # Handle clear button
    if clear_button:
        st.session_state.chat_history = []
        st.rerun()
    
    # Handle send button
    if send_button and user_input.strip():
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Get LLM response
        with st.spinner("🔄 Thinking..."):
            try:
                llm = ChatOpenAI(
                    model="arcee-ai/trinity-mini:free",
                    openai_api_key=OPENROUTER_API_KEY,
                    openai_api_base="https://openrouter.ai/api/v1",
                    temperature=0.7,
                    max_tokens=2000
                )
                
                # Build messages for the LLM
                messages = [
                    SystemMessagePromptTemplate.from_template(system_prompt).format(),
                ]
                
                # Add chat history
                for msg in st.session_state.chat_history[:-1]:  # Exclude current message temporarily
                    if msg["role"] == "user":
                        messages.append({"role": "user", "content": msg["content"]})
                    else:
                        messages.append({"role": "assistant", "content": msg["content"]})
                
                # Add current user message
                messages.append({"role": "user", "content": user_input})
                
                # Get response
                response = llm.invoke(messages)
                assistant_message = response.content if hasattr(response, 'content') else str(response)
                
                # Add assistant response to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": assistant_message
                })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error communicating with LLM: {str(e)}")
                st.session_state.chat_history.pop()  # Remove the failed user message
    
    # Display message count
    st.markdown("---")
    st.caption(f"📊 Messages in conversation: {len(st.session_state.chat_history)}")
def generate_layman_explanation_with_formula(layman_chain, explanation):
    """Return layman-friendly explanation using LLM"""
    pathway_summary = "\n".join([f"{p['rank']}. {p['path']}" for p in explanation.get("pathways", [])])
    
    return layman_chain.invoke({
        "drug_a": explanation["drug_a"],
        "drug_b": explanation["drug_b"],
        "drug_a_formula": explanation.get("drug_a_formula", "N/A"),
        "drug_b_formula": explanation.get("drug_b_formula", "N/A"),
        "prediction": explanation["prediction"],
        "confidence": round(explanation["confidence"], 4),
        "pathways": pathway_summary or "No pathway information available."
    })
