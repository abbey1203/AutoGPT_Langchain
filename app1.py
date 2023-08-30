
import os 
from apikey import apikey 

import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

os.environ['OPENAI_API_KEY'] = apikey

# App framework
st.title('🎼 Music GPT Generator 🎸')
prompt = st.text_input('Write a topic for your song') 

# Prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'], 
    template='write me a song name with {topic}'
)

lyric_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research'], 
    template='write me a song lyric based on this title TITLE: {title} while leveraging this wikipedia reserch:{wikipedia_research} '
)

# Memory 
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
lyric_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')


# Llms
llm = OpenAI(temperature=0.9) 
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
lyric_chain = LLMChain(llm=llm, prompt=lyric_template, verbose=True, output_key='lyric', memory=lyric_memory)

wiki = WikipediaAPIWrapper()

# Show stuff to the screen if there's a prompt
if prompt: 
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt) 
    lyric = lyric_chain.run(title=title, wikipedia_research=wiki_research)

    st.write(title) 
    st.write(lyric) 

    with st.expander('Title History'): 
        st.info(title_memory.buffer)

    with st.expander('Lyric History'): 
        st.info(lyric_memory.buffer)

    with st.expander('Wikipedia Research'): 
        st.info(wiki_research)
