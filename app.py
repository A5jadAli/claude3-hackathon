import os
import requests
import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import WikipediaAPIWrapper

st.title('🦜🔗 Blog Generator with Claude 3')
# Function to check if Anthropic API key is provided
def check_api_key():
    if 'ANTHROPIC_API_KEY' not in os.environ:
        st.stop()

# App framework
st.sidebar.title('Settings')
anthropic_api_key = st.sidebar.text_input('Anthropic API Key', '')

# Set Anthropic API key
if anthropic_api_key.startswith('sk-'):
    os.environ['ANTHROPIC_API_KEY'] = anthropic_api_key
    st.info("Anthropic API Key set successfully!")

# Check if Anthropic API key is provided
check_api_key()

prompt = st.text_input('Give me a topic to write a blog post:')

# Prompt templates
title_template = PromptTemplate(
    input_variables=['topic'],
    template="Generate a catchy and unique blog title about {topic}. Ensure it grabs attention and stands out."
)

content_template = PromptTemplate(
    input_variables=['title', 'wikipedia_research'],
    template="Write a comprehensive blog post titled '{title}' while leveraging this Wikipedia research {wikipedia_research} within 500 words. Cover key points, provide insights, and engage the reader. Also make sure wherever code is required to be written, it is written in Python, with proper Python markdown formatting."
)

# Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
content_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# LLMs
llm = ChatAnthropic(temperature=0.0, model_name='claude-3-opus-20240229')
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
content_chain = LLMChain(llm=llm, prompt=content_template, verbose=True, output_key='content', memory=content_memory)

wiki = WikipediaAPIWrapper()

# Show stuff to the screen if there's a prompt
if prompt:
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    content = content_chain.run(title=title, wikipedia_research=wiki_research)

    st.write(title)
    st.write(content)

    with st.expander('Title History'):
        st.info(title_memory.buffer)

    with st.expander('Content History'):
        st.info(content_memory.buffer)

    with st.expander('Wikipedia Research'):
        st.info(wiki_research)
