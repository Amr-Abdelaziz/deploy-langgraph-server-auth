"""
Chat App for Langgraph Agents using Streamlit.

This is a basic chat app/UI for interacting with Langgraph Agents via the Langgraph Server API. The app allows you to connect any Langgraph agents to a web UI, manage conversations, stream responses, and more.

This is a great starting point for learning how to build a full-stack AI application.
"""
import streamlit as st
from ai_launchpad.langgraph_module.frontends.streamlit_ui.api import (
    get_assistants,
    create_thread,
    search_threads,
    get_thread_state,
    run_thread_stream,
    delete_thread,
)
import json


#################################
# Session State Management
#################################

def initialize_session_state(user_id: str):
    """
    Initialize the session state with the user ID and other data we need to manage the chat app. Setting the user_id allows us to tag conversations to a user and manage conversations across sessions.

    Args:
        user_id (str): The user ID
    """
    if "user_id" not in st.session_state:
        st.session_state.user_id = user_id
    if "assistants" not in st.session_state:
        assistants_list = get_assistants()
        # Store assistants as a dict of {name: id} for easy lookup
        st.session_state.assistants = {assistant["name"]: assistant["assistant_id"] for assistant in assistants_list}
    if "active_assistant_id" not in st.session_state:
        st.session_state.active_assistant_id = list(st.session_state.assistants.values())[0]
    if "thread_ids" not in st.session_state:
        st.session_state.thread_ids = []
        threads = search_threads(st.session_state.user_id)
        for thread in threads:
            st.session_state.thread_ids.append(thread["thread_id"])
    if "selected_thread_id" not in st.session_state:
        if st.session_state.thread_ids:
            st.session_state.selected_thread_id = st.session_state.thread_ids[-1]  # newest thread
        else:
            st.session_state.selected_thread_id = None
    if "thread_state" not in st.session_state:
        st.session_state.thread_state = {}


def create_new_thread(user_id: str):
    """
    Create a new thread and update the session thread state to reflect the new thread.

    Args:
        user_id (str): The user ID
    """
    thread = create_thread(user_id)
    st.session_state.thread_ids.append(thread["thread_id"])
    st.session_state.thread_state = get_thread_state(thread["thread_id"])
    # Select the new thread
    st.session_state.selected_thread_id = thread["thread_id"]
    st.rerun()


def delete_thread_and_update_state(thread_id: str):
    """
    Delete a thread and update the session state to reflect the deleted thread.

    Args:
        thread_id (str): The thread ID
    """
    delete_thread(thread_id)
    st.session_state.thread_ids.remove(thread_id)
    # Clear thread state since we deleted the current thread
    st.session_state.thread_state = {}
    st.rerun()


initialize_session_state(user_id="kenny")


#################################
# UI
#################################

with st.sidebar:
    st.write("User ID: " + st.session_state.user_id)

    assistant = st.selectbox("Select Assistant", list(st.session_state.assistants.keys()))
    st.session_state.active_assistant_id = st.session_state.assistants[assistant]

    st.title("Conversations")

    if st.button("Create New Conversation"):
        create_new_thread(user_id=st.session_state.user_id)

    if st.session_state.thread_ids:
        def _on_select_thread():
            # Callback to load the thread state when a new thread is selected
            st.session_state.thread_state = get_thread_state(st.session_state.selected_thread_id)

        # Set default if no selection exists
        if "selected_thread_id" not in st.session_state or st.session_state.selected_thread_id not in st.session_state.thread_ids:
            st.session_state.selected_thread_id = st.session_state.thread_ids[-1]  # newest thread

        st.radio(
            "Select Conversation",
            options=st.session_state.thread_ids,
            format_func=lambda tid: tid[:8],
            key="selected_thread_id",
            on_change=_on_select_thread,
        )

    if st.button("Delete Conversation", type="primary"):
        if st.session_state.selected_thread_id:
            delete_thread_and_update_state(st.session_state.selected_thread_id)


st.title(f"Chatting with {assistant}")


# The thread_state already tracks the messages, so we just need to display them
if st.session_state.selected_thread_id and st.session_state.selected_thread_id in st.session_state.thread_ids:
    st.session_state.thread_state = get_thread_state(st.session_state.selected_thread_id)

# Display chat messages from the thread_state on app rerun
if st.session_state.thread_state:
    for message in st.session_state.thread_state["values"].get("messages", []):
        # Apply some formatting depending on the message type
        if message["type"] == "tool":
            with st.expander(f"🛠️ {message["name"]} < RESULTS > "):
                try:
                    json_message = json.loads(message["content"])
                    st.json(json_message)
                except:
                    st.write(message["content"])
        elif message["type"] == "ai" and message["tool_calls"]:
            with st.chat_message("ai"):
                st.markdown(f"🛠️ {message['tool_calls'][0]["name"]} < CALL >")
                st.json(message['tool_calls'][0]["args"])
        else:
            with st.chat_message(message["type"]):
                st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Send a message..."):
        # The chat messages here are for immediate display in the UI.
        # On the next render, all messages will be loaded from the thread_state

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            stream = run_thread_stream(st.session_state.active_assistant_id, st.session_state.selected_thread_id, {"messages": [prompt]})
            response = st.write_stream(stream)

        # Rerun the app to load the new messages from the thread_state
        st.rerun()
else:
    st.write("Create a new conversation to start chatting...")

# At the very bottom of every chat, we'll include the full session state for debugging
# Expand this section in the UI to inspect the full state of the app at any point
with st.expander("<DEBUG> Session State"):
    st.write(st.session_state)