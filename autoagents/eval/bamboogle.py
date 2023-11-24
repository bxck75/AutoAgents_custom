import glob
import json
import os
import pprint
import shutil

from autoagents.data.dataset import BAMBOOGLE
from autoagents.eval.metrics import get_common_stats
from autoagents.eval.hotpotqa.eval_async import check_answer_equivalency
from autoagents.agents.utils.constants import LOG_SAVE_DIR
from tqdm import tqdm
from langchain.chat_models import ChatOpenAI

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




async def eval(eval_results_path: str=LOG_SAVE_DIR):
    files = glob.glob(f"{eval_results_path}/*.json")
    """     evalllm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_organization=os.getenv("OPENAI_API_ORG"),
        temperature=0,
        model="gpt-3.5-turbo",
        request_timeout=120
    ) """
    evalllm = llm
    print(f"Found {len(files)} log files! Starts to analyze......")
    common_stats = get_common_stats(files)
    print(common_stats)
    accuracy = 0
    correct_res_dir, wrong_res_dir, err_res_dir = f"{eval_results_path}-eval/correct", f"{eval_results_path}-eval/wrong", f"{eval_results_path}-eval/error"
    os.makedirs(correct_res_dir, exist_ok=True)
    os.makedirs(wrong_res_dir, exist_ok=True)
    os.makedirs(err_res_dir, exist_ok=True)
    for file in tqdm(files):
        finish = False
        with open(file, "r") as f:
            log_data = json.load(f)
            has_error = any([True if "error" in entry else False for entry in log_data])
            for entry in log_data:
                if not has_error:
                    if "goal" in entry:
                        question = entry["goal"]
                    if "conversations" in entry:
                        output = json.loads(entry["conversations"][-1]["value"])
                        if output["action"] == "Tool_Finish":
                            finish = True
                            action_input = output["action_input"]
                            for i in range(len(BAMBOOGLE["questions"])):
                                if question == BAMBOOGLE["questions"][i]:
                                    answer = BAMBOOGLE["answers"][i]
                                    resp_obj = await check_answer_equivalency(question, answer, action_input, evalllm)
                                    is_correct = int(resp_obj.get("is_inferable", 0))
                                    if is_correct:
                                        shutil.copy2(file, correct_res_dir)
                                    else:
                                        shutil.copy2(file, wrong_res_dir)
                                    accuracy += is_correct
                else:
                    shutil.copy2(file, err_res_dir)
            if not finish:
                shutil.copy2(file, wrong_res_dir)
    counts = common_stats["counts"]
    total_samples = counts["total_samples"]
    finished_samples = counts["finished_samples"]
    print(f'accuracy overall is {accuracy}/{total_samples}={accuracy/total_samples}')
    print(f'accuracy on finished samples is {accuracy}/{finished_samples}={accuracy/finished_samples}')
    counts["accuracy on finished samples"] = accuracy/finished_samples
    counts["accuracy"] = accuracy/total_samples
    counts["average_answer_missing"] = (total_samples - finished_samples) / total_samples
    pprint.pprint(common_stats)
    with open(f"{eval_results_path}-eval/stats.json", "w") as f:
        json.dump(common_stats, f)
