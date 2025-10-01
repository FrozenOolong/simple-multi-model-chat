# app.py
import streamlit as st
from anthropic import Anthropic
from openai import OpenAI
import google.generativeai as genai
import json
from datetime import datetime
import os
from utils import ModelConfig
from dotenv import load_dotenv

load_dotenv()

class ChatbotApp:
    def __init__(self):
        # Initialize Chatbot Client
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.anthropic = Anthropic(api_key=anthropic_api_key)

        openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai = OpenAI(api_key=openai_api_key)

        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        self.deepseek = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

        gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.gemini = genai.configure(api_key=gemini_api_key)

        # Load configurations first (needed for conversation loading)
        self.load_saved_configurations()

        # Load recent conversations
        self.load_recent_conversations()

        # Initialize model configurations from utils
        self.providers = ModelConfig.get_available_providers()
        self.models = ModelConfig.get_available_models()
        self.model_descriptions = ModelConfig.get_model_descriptions()
        self.default_model = ModelConfig.get_default_model()
        
        
        # Initialize session state variables
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'current_provider' not in st.session_state:
            st.session_state.current_provider = "Anthropic"  #Default provider
        if 'current_model' not in st.session_state: # Default model if user didn't choose one
            st.session_state.current_model = self.default_model
        if 'custom_configs' not in st.session_state:
            st.session_state.custom_configs = {}  # {name: {"prompt": prompt, "model": model}}
        if 'conversation_saved' not in st.session_state:
            st.session_state.conversation_saved = False

    def get_current_model_display_name(self):
        """Get the display name of the current model or configuration."""
        if st.session_state.get('current_prompt'):
            for config_name, config in st.session_state.custom_configs.items():
                if (config["model"] == st.session_state.current_model and 
                    config["prompt"] == st.session_state.current_prompt):
                    return f"Custom Chatbot: {config_name}"
        
        # If no configuration match, find the model name
        for model_name, model_id in self.models.items():
            if model_id == st.session_state.current_model:
                return model_name
        
        return "Claude"  # Fallback name

    def new_conversation(self):
        """Start a new conversation by clearing chat history."""
        if st.session_state.messages and not st.session_state.conversation_saved:
            # Auto-save current conversation before clearing
            self.auto_save_conversation()
        
        # Clear the chat history and system prompt
        st.session_state.messages = []
        if 'current_prompt' in st.session_state:
            del st.session_state.current_prompt
        st.session_state.conversation_saved = False
        
        st.success("Started new conversation")
        st.rerun()
    
    def generate_conversation_name(self):
        """Generate a meaningful name for the conversation based on its content."""
        if len(st.session_state.messages) < 2:
            return None
            
        system_prompt = """Please analyze this conversation and create a brief (2-4 words), descriptive filename-safe name that captures its main topic or purpose. 
        Use only alphanumeric characters, underscores, and hyphens. No spaces. Example formats: 'data_analysis_help', 'python_debugging', 'machine_learning_basics'"""
        
        recent_messages = st.session_state.messages[-10:]
        
        try:
            response = self.anthropic.messages.create(
                model="claude-3-haiku-20240307",
                system=system_prompt,
                messages=[
                    {"role": "user", "content": f"Generate a name for this conversation: {str(recent_messages)}"}
                ],
                max_tokens=50
            )
            
            conversation_name = response.content[0].text.strip()
            conversation_name = ''.join(c for c in conversation_name if c.isalnum() or c in '-_')
            return conversation_name
        except Exception as e:
            st.error(f"Error generating conversation name: {str(e)}")
            return None

    def auto_save_conversation(self):
        """Automatically save the conversation after each message exchange."""
        if not st.session_state.messages or st.session_state.conversation_saved:  
            return
        
        save_dir = "conversations"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Generate a conversation name
        conv_name = self.generate_conversation_name()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Use the generated name or fallback to timestamp
        if conv_name:
            filename = f"{conv_name}_{timestamp}.json"
        else:
            filename = f"conversations_{timestamp}.json"
            
        filepath = os.path.join(save_dir, filename)
        
        # Prepare conversation data
        conversation_data = {
            "messages": st.session_state.messages,
            "model": st.session_state.current_model,
            "custom_configs": st.session_state.custom_configs,
            "timestamp": timestamp,
            "name": conv_name if conv_name else "Untitled Conversation"
        }
        
        # Save the conversation
        with open(filepath, "w") as f:
            json.dump(conversation_data, f, indent=2)
            
        # Store the latest filepath in session state
        st.session_state.latest_save = filepath
        st.session_state.conversation_saved = True
        
        # Update conversation list without showing success message
        self.list_saved_conversations()

    def save_conversation(self):
        """Manual save with user notification."""
        if not st.session_state.messages:  # Don't save empty conversations
            st.sidebar.warning("No messages to save")
            return
            
        self.auto_save_conversation()
        if hasattr(st.session_state, 'latest_save'):
            st.sidebar.success(f"Conversation saved to: {os.path.basename(st.session_state.latest_save)}")

    def list_saved_conversations(self):
        """Display list of saved conversations in sidebar."""
        save_dir = "conversations"
        if os.path.exists(save_dir):
            saved_files = sorted(
                [f for f in os.listdir(save_dir) if f.endswith('.json')],
                reverse=True
            )
            if saved_files:
                st.sidebar.header("Recent Conversations")
                for file in saved_files[:10]:  # Show last 10 conversations
                    if st.sidebar.button(f"Load: {file}"):
                        self.load_conversation(os.path.join(save_dir, file))

    def load_conversation(self, filepath):
        """Load a conversation from a file."""
        try:
            if isinstance(filepath, str):
                with open(filepath, 'r') as f:
                    conversation_data = json.load(f)
            else:  # Handle uploaded file
                conversation_data = json.load(filepath)
                
            st.session_state.messages = conversation_data["messages"]
            st.session_state.current_model = conversation_data["model"]
            st.session_state.custom_configs = conversation_data.get("custom_configs", {})
            st.sidebar.success("Conversation loaded successfully")
        except Exception as e:
            st.sidebar.error(f"Error loading conversation: {str(e)}")
    
    def load_recent_conversations(self):
        """Load recent conversations when app starts."""
        save_dir = "conversations"
        if os.path.exists(save_dir):
            # Get all json files with their creation times
            files_with_times = []
            for filename in os.listdir(save_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(save_dir, filename)
                    try:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                            timestamp = data.get('timestamp', 
                                        datetime.fromtimestamp(os.path.getctime(filepath)).strftime("%Y%m%d_%H%M%S"))
                            files_with_times.append((filename, timestamp, filepath))
                    except Exception as e:
                        st.error(f"Error reading file {filename}: {str(e)}")
                        continue

            # Sort by timestamp in descending order
            files_with_times.sort(key=lambda x: x[1], reverse=True)
            
            # # Store recent conversations in session state
            st.session_state.recent_conversations = files_with_times[:5]
    
    def load_saved_configurations(self):
        """Load saved configurations from file."""
        config_dir = "custom_configs"
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
            
        if 'custom_configs' not in st.session_state:
            st.session_state.custom_configs = {}
            
        # Load each configuration file
        for filename in os.listdir(config_dir):
            if filename.endswith('.json'):
                config_name = filename[:-5]  # Remove .json extension
                filepath = os.path.join(config_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        config_data = json.load(f)
                    st.session_state.custom_configs[config_name] = config_data
                except Exception as e:
                    st.error(f"Error loading configuration {config_name}: {str(e)}")


    def save_configuration(self, config_name, config_data):
        """Save configurations to file."""
        config_dir = "custom_configs"
        filepath = os.path.join(config_dir, f"{config_name}.json")
        try:
            with open(filepath, 'w') as f:
                json.dump(config_data, f, indent=2)
        except Exception as e:
            st.error(f"Error saving configuration {config_name}: {str(e)}")

    def delete_configuration(self, config_name):
        """Delete a configuration and its file."""
        config_dir = "custom_configs"
        filepath = os.path.join(config_dir, f"{config_name}.json")
        
        # Remove from session state and delete file
        if config_name in st.session_state.custom_configs:
            del st.session_state.custom_configs[config_name]
        
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            st.error(f"Error deleting configuration file {config_name}: {str(e)}")
    

    def setup_sidebar(self):
        """Setup the sidebar with all configuration options."""
        with st.sidebar:
            if st.button("New Conversation", key="new_conv_top"):
                self.new_conversation()

            # Model Selection
            st.header("Chatbot Selection")
            # Create combined options list
            model_options = list(self.models.keys())
            config_options = list(st.session_state.get('custom_configs', {}).keys())
            all_options = model_options + [f"Custom Chatbot: {config}" for config in config_options]
            
            # Find current selection
            if 'current_model' not in st.session_state:
                st.session_state.current_model = list(self.models.values())[0]
            if 'current_prompt' not in st.session_state:
                st.session_state.current_prompt = None

            # Selection dropdown
            name = 'model_select_box'
            selected_option = st.selectbox(
                "Choose Model or Custom Chatbot",
                options=all_options,
                index=all_options.index(st.session_state.get(name, all_options[0])),
                on_change=lambda: st.session_state.update({name: st.session_state[name + "_key"]}),
                key=name + "_key"
            )
            
            # Handle selection
            if selected_option.startswith("Custom Chatbot:"):
                # Handle custom configuration selection
                config_name = selected_option[15:].strip()  # Remove "Custom Chatbot: " prefix
                config = st.session_state.custom_configs[config_name]
                st.session_state.current_model = config["model"]
                st.session_state.current_prompt = config["prompt"]
                
                # Display configuration details
                st.markdown("**Configuration Details:**")
                st.markdown(f"Model: {next(name for name, model in self.models.items() if model == config['model'])}")
                with st.expander("System Prompt"):
                    st.text_area("", value=config["prompt"], height=100, disabled=True, key="config_prompt_display")
                # st.markdown("System Prompt:")
                # st.text_area("", value=config["prompt"], height=100, disabled=True, key="config_prompt_display")
                model_name = ModelConfig.get_model_name_from_id(config["model"])
                st.session_state.current_provider = ModelConfig.get_model_provider_from_name(model_name)
            else:
                # Handle standard model selection
                st.session_state.current_model = self.models[selected_option]
                st.session_state.current_prompt = None

                st.session_state.current_provider = ModelConfig.get_model_provider_from_name(selected_option)

                # Display model information
                st.markdown("**Model Description:**")
                st.write(self.model_descriptions[selected_option])
            
            st.divider()

            # Custom configurations management
            st.header("Custom Chatbot")
            
            # Create new configuration
            with st.expander("Create New Custom Chatbot"):
                config_name = st.text_input("Chatbot Name")
                selected_config_model = st.selectbox(
                    "Select Model",
                    list(self.models.keys()),
                    key="config_model_select"
                )
                system_prompt = st.text_area("System Prompt")
                
                if st.button("Save Configuration"):
                    if config_name and system_prompt:
                        config_data = {
                            "prompt": system_prompt,
                            "model": self.models[selected_config_model]
                        }
                        st.session_state.custom_configs[config_name] = config_data
                        self.save_configuration(config_name, config_data)  # Save individual config file
                        st.success(f"Saved configuration: {config_name}")
                        st.rerun()
            
            # Delete existing configuration
            if st.session_state.custom_configs:
                st.subheader("Delete Configuration")
                config_to_delete = st.selectbox(
                    "Select Configuration to Delete",
                    ["None"] + list(st.session_state.custom_configs.keys()),
                    key="config_delete_select"
                )
                
                if config_to_delete != "None" and st.button("Delete Selected Configuration"):
                    del st.session_state.custom_configs[config_to_delete]
                    if st.session_state.get('current_prompt'):  # If currently using a config
                        st.session_state.current_prompt = None  # Reset to standard model
                    self.delete_configuration(config_name)
                    st.success(f"Deleted configuration: {config_to_delete}")
                    st.rerun()
            
            st.divider()
            # Always show recent conversations at the top
            if hasattr(st.session_state, 'recent_conversations'):
                st.header("Recent Conversations")
                for filename, timestamp, filepath in st.session_state.recent_conversations:
                    # Convert timestamp to readable format
                    try:
                        dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                        readable_time = dt.strftime("%Y-%m-%d %H:%M")
                    except:
                        readable_time = timestamp
                    
                    if st.button(f"{readable_time}: {filename}", key=f"load_{filename}"):
                        self.load_conversation(filepath)
                        self.load_recent_conversations()  # Refresh the list after loading
            
            # Conversation Management
            st.header("Conversation Management")
            if st.button("Save Conversation"):
                self.save_conversation()
            
            uploaded_file = st.file_uploader("Load Conversation", type="json")
            if uploaded_file is not None:
                self.load_conversation(uploaded_file)

    def setup_chat_interface(self):
        """Setup the main chat interface."""
        current_name = self.get_current_model_display_name()
        st.title(f"Chat with {current_name}")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input():
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.conversation_saved = False  # Mark conversation as modified

            with st.chat_message("user"):
                st.write(prompt)
            
            # Prepare messages for API call
            messages = []
            if st.session_state.get('current_prompt'):
                messages.append({
                    "role": "system",
                    "content": st.session_state.current_prompt
                })
            
            messages.extend(st.session_state.messages)
            
            # Get Claude's response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    api_params = {
                        "model": st.session_state.current_model,
                        "messages": st.session_state.messages,
                        "max_tokens": 1024
                        }
                    # Add system prompt if it exists
                    if st.session_state.get('current_prompt'):
                        api_params["system"] = st.session_state.current_prompt
                    
                    # Modify message generation by provider
                    if st.session_state.current_provider=="Anthropic":
                        response = self.anthropic.messages.create(**api_params)
                        response_content = response.content[0].text
                    elif st.session_state.current_provider=="OpenAI":
                        response = self.openai.chat.completions.create(**api_params)
                        response_content = response.choices[0].message.content
                    elif st.session_state.current_provider=="DeepSeek":
                        response = self.deepseek.chat.completions.create(**api_params)
                        response_content = response.choices[0].message.content
                    elif st.session_state.current_provider=="Google":
                        model = self.gemini.GenerativeModel(st.session_state.current_model)
                        response = model.generate_content(st.session_state.messages)
                    
                    st.write(response_content)
 
            # Add assistant's response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response_content})

    def run(self):
        """Run the chatbot application."""
        self.setup_sidebar()
        self.setup_chat_interface()

if __name__ == "__main__":
    app = ChatbotApp()
    app.run()
