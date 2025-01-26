import os
import streamlit as st
from langchain.llms import OpenAI
from langchain import LLMChain, PromptTemplate

# Streamlit app
st.title("Generate Custom Facebook with LangChain + OpenAI")

# Step 1: Prompt user for OpenAI API key
openai_api_key = st.text_input("Enter your OpenAI API key", type="password")
if not openai_api_key:
    st.warning("Please enter your OpenAI API key.")
    st.stop()  # Stop execution until key is provided

# Set the OPENAI_API_KEY environment variable so that LangChain can use it
os.environ["OPENAI_API_KEY"] = openai_api_key

# Step 2: Prompt user for topic
topic = st.text_input("Enter the topic for your facebook post", "RCM latest trends")

# Step 3: Prompt user for number of tweets
num_posts = st.number_input(
    "Number of facebook post to generate:",
    min_value=1,
    max_value=10,
    value=5
)

# Create a PromptTemplate to instruct the LLM
prompt = PromptTemplate(
    input_variables=["topic", "num_posts"],
    template="""
You are a RCM Manager and generally share the useful information, tips, trends and solutions. 
Generate {num_posts} creative facebook post about the topic, give some information
or statistic or tips. Which should make 
user to engage or encourage to give their input"{topic}".

Requirements:
1. Each Post must be greater than 1000 and fewer than 5000 characters.
2. Each post shold be useful to user and very structured
3. Dont sound as marketing.
4. Put Hashtag as well.

Return the Post in plain text.
"""
)

# Instantiate the LLM (OpenAI) through LangChain
llm = OpenAI(
    temperature=0.7,   # Adjust for creativity
    max_tokens=500,    # Enough tokens to handle multiple tweets
)

# Build the LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# Button to trigger tweet generation
if st.button("Generate Facebook Post"):
    with st.spinner("Generating Post..."):
        response = chain.run(topic=topic, num_posts=str(num_posts))
    st.write("### Generated Posts:")
    st.write(response)
