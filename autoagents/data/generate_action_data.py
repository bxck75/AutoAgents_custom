# Script generates action data from goals calling GPT-4
import os
import asyncio
import argparse
from tqdm import tqdm

from multiprocessing import Pool

from autoagents.agents.agents.search import ActionRunner
from autoagents.eval.test import AWAIT_TIMEOUT
from langchain.chat_models import ChatOpenAI
import json


from dotenv import load_dotenv, find_dotenv
from typing import Union
from huggingface_hub import login
from langchain.llms import HuggingFaceHub
from langchain.chains import  LLMChain
from elevenlabs import set_api_key
import warnings
import re
warnings.filterwarnings('ignore')
from langchain.tools import tool
from langchain.tools import Tool
from langchain.agents import AgentType, initialize_agent
from langchain.chains import LLMMathChain,LLMChain
from langchain.prompts import  PromptTemplate
from langchain.schema import SystemMessage
from langchain.document_loaders import YoutubeLoader
from langchain_experimental.autonomous_agents import HuggingGPT
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import load_prompt, ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts import PromptTemplate
from transformers import load_tool
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import VectorDBQA
from langchain.document_loaders import TextLoader

from operator import itemgetter

from langchain.prompts import ChatPromptTemplate

from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from langchain.memory import CombinedMemory, ConversationBufferMemory, ConversationSummaryMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers.tavily_search_api import TavilySearchAPIRetriever
from langchain.schema.messages import SystemMessage

from dotenv import load_dotenv, find_dotenv
from huggingface_hub import login
from langchain.llms import HuggingFaceHub
from langchain.chains import  LLMChain
from elevenlabs import set_api_key
from langchain.agents.agent import AgentExecutor, AgentOutputParser
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.tools import tool
from langchain.tools import Tool
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from langchain.agents import AgentType, initialize_agent
from langchain.chains import LLMMathChain,LLMChain
from langchain.prompts import  PromptTemplate
from langchain.schema import SystemMessage
from langchain.document_loaders import YoutubeLoader
from langchain_experimental.autonomous_agents import HuggingGPT
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import load_prompt, ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts import PromptTemplate
from transformers import load_tool
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import load_prompt
from langchain.chains import VectorDBQA
from langchain.document_loaders import TextLoader
from operator import itemgetter
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from langchain.memory import CombinedMemory, ConversationBufferMemory, ConversationSummaryMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers.tavily_search_api import TavilySearchAPIRetriever
from langchain.schema.messages import SystemMessage
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS
from tempfile import TemporaryDirectory
from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
import argparse
from langchain.schema.runnable import (
    ConfigurableField,
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.vectorstores import FAISS
from langchain.document_loaders import WebBaseLoader
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)
import getpass
import os
import sys
import pandas as pd
import uuid
import faiss
load_dotenv(find_dotenv())
#CREDENTIALS
from credits import (
    HUGGINGFACE_TOKEN,
    HUGGINGFACE_TOKEN as HUGGINGFACEHUB_API_TOKEN,
    HUGGINGFACE_EMAIL,
    HUGGINGFACE_PASS,
    ELEVENLABS_API_KEY,
    SERPAPI_API_KEY)

from langchain.utilities import PythonREPL
from langchain.utilities.serpapi import SerpAPIWrapper
serp_search = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
from langchain.tools import ElevenLabsText2SpeechTool
from langchain.retrievers.document_compressors.embeddings_filter import EmbeddingsFilter
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.prompts.chat import (
                                ChatPromptTemplate,
                                HumanMessagePromptTemplate,
                            )
set_api_key(ELEVENLABS_API_KEY)
script_dir = os.path.dirname(os.path.abspath(__file__))

tts = ElevenLabsText2SpeechTool(eleven_api_key=ELEVENLABS_API_KEY, voice="amy",)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

repo_list = [
    {"user": "tiiuae", "model": "falcon-7b-instruct"},
    {"user": "mistralai", "model": "Mistral-7B-v0.1"},
    {"user": "microsoft", "model": "Orca-2-7b"},
    {"user": "codellama", "model": "CodeLlama-7b-Python-hf"},
]
repo_ids = []  # Initialize an empty list to store repo IDs
for i, repo in enumerate(repo_list):
    repo_id = f"{repo['user']}/{repo['model']}"
    repo_ids.append(repo_id)
    # add repos to memory

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
os.environ["HUGGINGFACE_EMAIL"] = HUGGINGFACE_EMAIL
os.environ["HUGGINGFACE_PASS"] = HUGGINGFACE_PASS

# contains the list of repo IDs
print(repo_ids)

# Login
login(HUGGINGFACEHUB_API_TOKEN)

class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

output_parser = CustomOutputParser()
################################### WORKING HF_LLM## ################################################

#llm = HuggingFaceHub(repo_id=repo_ids[0], model_kwargs={"temperature": 0.1, "max_new_tokens": 1200})
llm = HuggingFaceHub(
                    repo_id=repo_ids[0], 
                    task="text-generation",
                    model_kwargs = {
                        "min_length": 200,
                        "max_length":2000,
                        "temperature":0.1,
                        "max_new_tokens":256,
                        "num_return_sequences":1
                    }
                )


################################### WORKING HF_LLM #################################################
print("Testing llm")

""" system_message = "You are Orca, an AI language model created by Microsoft. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."
user_message = "How can you determine if a restaurant is popular among locals or mainly attracts tourists, and why might this information be useful?"

# mistral
mistral_prompt = f"<s>[PRE] {system_message} [/PRE] [INST] {user_message} [/INST]"

# llama
llama_prompt = f'''
    [INSTRUCTION] {user_message} [/INSTRUCTION]
    [CONTEXT] {system_message} [/CONTEXT]
    '''

# orca
microsoft_orca_prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant"
microsoft_orca_prompt1 = f"<|im_start|>system\n{system_message}\nuser\n{user_message}<|im_end|>\n"
 """
########################################################################################################
# We'll make a temporary directory to avoid clutter
working_directory = TemporaryDirectory()

# Construct the agent. We will use the default agent type here.
# See documentation for a full list of options.

data_dir="./data"
embeddings= HuggingFaceEmbeddings()




async def work(user_input):
    outputq = asyncio.Queue()
    """ llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"),
                     openai_organization=os.getenv("OPENAI_API_ORG"),
                     temperature=0.,
                     model_name="gpt-4") """
    llm = llm
    runner = ActionRunner(outputq, llm=llm, persist_logs=True)
    task = asyncio.create_task(runner.run(user_input, outputq))

    while True:
        try:
            output = await asyncio.wait_for(outputq.get(), AWAIT_TIMEOUT)
        except asyncio.TimeoutError:
            return
        if isinstance(output, RuntimeWarning):
            print(output)
            continue
        elif isinstance(output, Exception):
            print(output)
            return
        try:
            parsed = json.loads(output)
            if parsed["action"] in ("Tool_Finish", "Tool_Abort"):
                break
        except:
            pass
    await task

def main(q):
    asyncio.run(work(q))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--goals', type=str, help="file containing JSON array of goals", required=True)
    parser.add_argument("--num_data", type=int, default=-1, help="number of goals for generation")
    args = parser.parse_args()
    with open(args.goals, "r") as file:
        data = json.load(file)
        if args.num_data > -1 and len(data) > args.num_data:
            data = data[:args.num_data]
    with Pool(processes=4) as pool:
        for _ in tqdm(pool.imap_unordered(main, data), total=len(data)):
            pass
