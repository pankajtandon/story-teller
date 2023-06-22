
from dotenv import find_dotenv, load_dotenv
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from transformers import pipeline
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from PIL import Image
import requests
from io import BytesIO


load_dotenv(find_dotenv())

PAGE_CONFIG = {"page_title": "Hello baby!", "page_icon": "smiley", "layout": "centered"}
st.set_page_config(**PAGE_CONFIG)
st.title("Welcome to our world of baby delights!")
st.subheader("We are head over heels!")


uploaded_file = st.file_uploader("Upload your image", type =  ['png', 'jpg'])

genre_choice = st.selectbox(
    label = 'Which genre should the story be in?',
    options= ['Scary', 'Funny', 'Horrific', 'Suspenseful', 'Poetic', 'Haiku Like', 'Pedantic', 'Shakesperean', 'Gothic', 'Romantic']
)

debug = st.checkbox("Would you like to see debug info?")

def story_teller(scenario, genre):
  template = """
  You are a great story teller! Based on the scenario below, tell me a story of about 25 words that is interesting and {genre}!
  Scenario: {scenario}
  """

  prompt = PromptTemplate(template = template, input_variables = ["scenario", "genre"])

  print("Prompt", prompt)

  story_teller_chain = LLMChain(llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature = 1), prompt = prompt, verbose = True)
  story = story_teller_chain.predict(scenario = scenario, genre = genre)
  return story


if (uploaded_file is not None):
  bytes = uploaded_file.getvalue()
  with open(uploaded_file.name, "wb") as file:
    file.write(bytes)
  
  st.image(uploaded_file, caption = "Uploaded Image", use_column_width = True)
  image_to_text = pipeline("image-to-text", model = "Salesforce/blip-image-captioning-base")
  # image_to_text = pipeline("image-to-text", model = "nlpconnect/vit-gpt2-image-captioning")

  text = image_to_text(uploaded_file.name)
  # text = image_to_text(img)

  print(text[0]["generated_text"])


  story = story_teller(scenario = text[0]["generated_text"], genre = genre_choice)

  st.write("Here's your ", genre_choice, " story")
  print(story)
  st.write(story)

