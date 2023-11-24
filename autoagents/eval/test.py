import argparse
import asyncio
import json
import os
from tqdm.asyncio import tqdm_asyncio

from autoagents.agents.agents.search import ActionRunner
from autoagents.agents.agents.wiki_agent import WikiActionRunner, WikiActionRunnerV3
from autoagents.agents.agents.search_v3 import ActionRunnerV3
from autoagents.agents.models.custom import CustomLLM, CustomLLMV3
from autoagents.agents.utils.constants import LOG_SAVE_DIR
from autoagents.data.dataset import BAMBOOGLE, DEFAULT_Q, FT, HF
from autoagents.eval.bamboogle import eval as eval_bamboogle
from autoagents.eval.hotpotqa.eval_async import HotpotqaAsyncEval, NUM_SAMPLES_TOTAL
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




OPENAI_MODEL_NAMES = {"gpt-3.5-turbo", "gpt-4"}
AWAIT_TIMEOUT: int = 120
MAX_RETRIES: int = 2


async def work(user_input: str, model: str, temperature: int, agent: str, prompt_version: str, persist_logs: bool, log_save_dir: str):
    if model not in OPENAI_MODEL_NAMES:
        if prompt_version == "v2":
            llm = CustomLLM(
                model_name=model,
                temperature=temperature,
                request_timeout=AWAIT_TIMEOUT
            )
        elif prompt_version == "v3":
            llm = CustomLLMV3(
                model_name=model,
                temperature=temperature,
                request_timeout=AWAIT_TIMEOUT
            )
    else:
            """         llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_organization=os.getenv("OPENAI_API_ORG"),
            temperature=temperature,
            model_name=model,
            request_timeout=AWAIT_TIMEOUT
        ) """
    llm = llm
    retry_count = 0
    while retry_count < MAX_RETRIES:
        outputq = asyncio.Queue()
        if agent == "ddg":
            if prompt_version == "v2":
                runner = ActionRunner(outputq, llm=llm, persist_logs=persist_logs)
            elif prompt_version == "v3":
                runner = ActionRunnerV3(outputq, llm=llm, persist_logs=persist_logs)
        elif agent == "wiki":
            if prompt_version == "v2":
                runner = WikiActionRunner(outputq, llm=llm, persist_logs=persist_logs)
            elif prompt_version == "v3":
                runner = WikiActionRunnerV3(outputq, llm=llm, persist_logs=persist_logs)
        task = asyncio.create_task(runner.run(user_input, outputq, log_save_dir))
        while True:
            try:
                output = await asyncio.wait_for(outputq.get(), AWAIT_TIMEOUT)
            except asyncio.TimeoutError:
                task.cancel()
                retry_count += 1
                break
            if isinstance(output, RuntimeWarning):
                print(f"Question: {user_input}")
                print(output)
                continue
            elif isinstance(output, Exception):
                task.cancel()
                print(f"Question: {user_input}")
                print(output)
                retry_count += 1
                break
            try:
                parsed = json.loads(output)
                print(json.dumps(parsed, indent=2))
                print("-----------------------------------------------------------")
                if parsed["action"] == "Tool_Finish":
                    return await task
            except:
                print(f"Question: {user_input}")
                print(output)
                print("-----------------------------------------------------------")
    

async def main(questions, args):
    sem = asyncio.Semaphore(10)
    
    async def safe_work(user_input: str, model: str, temperature: int, agent: str, prompt_version: str, persist_logs: bool, log_save_dir: str):
        async with sem:
            return await work(user_input, model, temperature, agent, prompt_version, persist_logs, log_save_dir)
    
    persist_logs = True if args.persist_logs else False
    await tqdm_asyncio.gather(*[safe_work(q, args.model, args.temperature, args.agent, args.prompt_version, persist_logs, args.log_save_dir) for q in questions])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="model to be tested")
    parser.add_argument("--temperature", type=float, default=0, help="model temperature")
    parser.add_argument("--agent",
        default="ddg",
        const="ddg",
        nargs="?",
        choices=("ddg", "wiki"),
        help='which action agent we want to interact with(default: ddg)'
    )
    parser.add_argument("--persist-logs", action="store_true", help="persist logs on disk, enable this feature for later eval purpose")
    parser.add_argument("--log-save-dir", type=str, default=LOG_SAVE_DIR, help="dir to save logs")
    parser.add_argument("--dataset",
        default="default",
        const="default",
        nargs="?",
        choices=("default", "hotpotqa", "ft", "hf", "bamboogle"),
        help='which dataset we want to interact with(default: default)'
    )
    parser.add_argument("--eval", action="store_true", help="enable automatic eval")
    parser.add_argument("--prompt-version",
        default="v2",
        const="v3",
        nargs="?",
        choices=("v2", "v3"),
        help='which version of prompt to use(default: v2)'
    )
    parser.add_argument("--slice", type=int, help="slice the dataset from left, question list will start from index 0 to slice - 1")
    args = parser.parse_args()
    print(args)
    if args.prompt_version == "v3" and args.model in OPENAI_MODEL_NAMES:
        raise ValueError("Prompt v3 is not compatiable with OPENAI models, please adjust your settings!")
    if not args.persist_logs and args.eval:
        raise ValueError("Please enable persist_logs feature to allow eval code to run!")
    if not args.log_save_dir and args.persist_logs:
        raise ValueError("Please endbale persist_logs feature to configure log dir location!")
    questions = []
    if args.dataset == "ft":
        questions = [q for _, q in FT]
    elif args.dataset == "hf":
        questions = [q for _, q in HF]
    elif args.dataset == "hotpotqa":
        hotpotqa_eval = HotpotqaAsyncEval(model=args.model)
        questions = hotpotqa_eval.get_questions(args.slice or NUM_SAMPLES_TOTAL)
    elif args.dataset == "bamboogle":
        questions = BAMBOOGLE["questions"]
    else:
        questions = [q for _, q in DEFAULT_Q]
    if args.slice and args.dataset != "hotpotqa":
        questions = questions[:args.slice]
    asyncio.run(main(questions, args))
    if args.eval:
        if args.dataset == "bamboogle":
            if args.log_save_dir:
                asyncio.run(eval_bamboogle(args.log_save_dir))
            else:
                asyncio.run(eval_bamboogle())
        elif args.dataset == "hotpotqa":
            if args.log_save_dir:
                hotpotqa_eval.run(args.log_save_dir)
            else:
                hotpotqa_eval.run()
