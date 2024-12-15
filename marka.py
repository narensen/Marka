import streamlit as st
from langchain_groq import ChatGroq

# Configure page layout
st.set_page_config(layout="wide")

# Main page content
st.title("Marketing and Sales AI Assistant")
st.write("Your main page content goes here.")

# Initialize session states
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# ChatGroq setup
groq_api_key = "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"
if groq_api_key:
    groq_chat = ChatGroq(
        api_key=groq_api_key,
        model_name="llama-3.2-90b-vision-preview",
        temperature=0.7
    )

# Chat interface on the main page
st.subheader("Chat with the AI Assistant")
user_question = st.text_input("Ask a Marketing or Sales question:", key="user_input_main")

if user_question:
    # Add user question to conversation history
    st.session_state.conversation_history.append({"role": "user", "content": user_question})
    
    # Construct the prompt
    prompt = f"""You are an AI Powered Chatbot who provides answers to Marketing and Sales related queries. 
    Your responses should always be confident and actionable. 
    You should not respond to any other kind of questions which are unrelated to Marketing and Sales.
    User question: {user_question}
    """
    
    # Generate the AI response
    try:
        response = groq_chat.invoke(st.session_state.conversation_history + [{"role": "user", "content": prompt}])
        ai_response = response.content
        # Add AI response to conversation history
        st.session_state.conversation_history.append({"role": "assistant", "content": ai_response})
    except Exception as e:
        ai_response = f"Error: {str(e)}"
        st.session_state.conversation_history.append({"role": "assistant", "content": ai_response})
    
    # Display the AI response
    st.markdown(f"**AI:** {ai_response}")

# Display chat history
if st.session_state.conversation_history:
    st.subheader("Conversation History")
    for message in st.session_state.conversation_history:
        if message['role'] == 'user':
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**AI:** {message['content']}")
