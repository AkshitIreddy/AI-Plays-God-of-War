from transformers import pipeline
import random
import json
from langchain.llms import Cohere
from langchain import PromptTemplate, LLMChain
import cv2

def llm_agent(screen):
    image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
    cv2.imwrite('temp.png', screen)
    file_description = image_to_text('temp.png')[0]['generated_text']
    # Enter an API key
    key=""
    # Initialise model
    llm = Cohere(cohere_api_key=key,
                 model='command-xlarge-beta', temperature=1.2, max_tokens=1700)
    # create the template string
    template = """Instructions:\nEnemy NPC near Main character, choose move light attack, heavy attack or dodge back.\nSituation:Monster with a sword near Kratos\nMove:light attack\nSituation:{file_description}\nMove:"""
    # create prompt
    prompt = PromptTemplate(template=template, input_variables=["file_description"])
    # Create and run the llm chain
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response = llm_chain.run(file_description=file_description)
    return response 