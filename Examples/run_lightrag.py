# lightrag_example.py
import asyncio
import logging
import os

import nest_asyncio
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.hf import hf_embed
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer

from shared_code import BASE_ARG_CONFIG, init_data, parse_args, setup_logging, process_corpus

# Apply nest_asyncio for Jupyter environments
nest_asyncio.apply()
logger = setup_logging("lightrag_processing.log")


SYSTEM_PROMPT = """
---Role---
You are a helpful assistant responding to user queries.

---Goal---
Generate direct and concise answers based strictly on the provided Knowledge Base.
Respond in plain text without explanations or formatting.
Maintain conversation continuity and use the same language as the query.
If the answer is unknown, respond with "I don't know".

---Conversation History---
{history}

---Knowledge Base---
{context_data}
"""

async def llm_model_func(
    prompt: str,
    system_prompt: str = None,
    history_messages: list = [],
    keyword_extraction: bool = False,
    **kwargs
) -> str:
    """LLM interface function using OpenAI-compatible API"""
    # Get API configuration from kwargs
    model_name = kwargs.get("model_name", "qwen2.5-14b-instruct")
    base_url = kwargs.get("base_url", "")
    api_key = kwargs.get("api_key", "")
    
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url=base_url,
        api_key=api_key,
        **kwargs
    )

async def initialize_rag(
    base_dir: str,
    source: str,
    mode:str,
    model_name: str,
    embed_model_name: str,
    llm_base_url: str,
    llm_api_key: str
) -> LightRAG:
    """Initialize LightRAG instance for a specific corpus"""
    working_dir = os.path.join(base_dir, source)
    
    # Create directory for this corpus
    os.makedirs(working_dir, exist_ok=True)
    
    if mode == "API":
        tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
        embed_model = AutoModel.from_pretrained(embed_model_name)
        # Initialize embedding function
        embedding_func = EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=lambda texts: hf_embed(texts, tokenizer, embed_model),
        )
        
        # Create LLM configuration
        llm_kwargs = {
            "model_name": model_name,
            "base_url": llm_base_url,
            "api_key": llm_api_key
        }

        llm_model_func_input = llm_model_func
    elif mode == "ollama":
        embedding_func = EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts, embed_model=embed_model_name, host=llm_base_url
            ),
        )

        llm_kwargs = {
            "host": llm_base_url,
            "options": {"num_ctx": 32768},
        }

        llm_model_func = ollama_model_complete
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'API' or 'ollama'.")
    
    # Create RAG instance
    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func,
        llm_model_name=model_name,
        llm_model_max_async=4,
        llm_model_max_token_size=32768,
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
        embedding_func=embedding_func,
        llm_model_kwargs=llm_kwargs
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

async def lightrag_init_func(corpus_name, context, base_dir, args):
    """
    Initialize LightRAG for a specific corpus
    Take:
    - corpus_name: Name of the corpus
    - context: Text content of the corpus
    - base_dir: Base directory for storing data
    - args: Parsed command-line arguments
    Return:
    - Initialized LightRAG instance
    """
    rag = await initialize_rag(
        base_dir=base_dir,
        source=corpus_name,
        mode=args.mode,
        model_name=args.model_name,
        embed_model_name=args.embed_model,
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key
    )
    await rag.insert(context)
    return rag

async def lightrag_query_func(rag, question, args) -> (str, list):
    """
    Query LightRAG with a question and return the response and context
    Take:
    - rag: Initialized LightRAG instance
    - question: Dictionary with 'question' key
    - args: Parsed command-line arguments
    Return:
    - response: Generated answer string
    - context: List of context documents used for the answer
    """
    query_param = QueryParam(
        mode='hybrid',
        top_k=args.retrieve_topk,
        max_token_for_text_unit=4000,
        max_token_for_global_context=4000,
        max_token_for_local_context=4000
    )
    response, context = rag.query(
        question["question"],
        param=query_param,
        system_prompt=SYSTEM_PROMPT
    )
    if asyncio.iscoroutine(response):
        response = await response
    return str(response), context

def main():
    """
        Main function to process corpora and answer questions using Method (LightRAG).
        This can be used as a template for other (Graph)RAG systems.
        The main function should not require changes beyond
        - adding commandline arguments and
        - changing the rag_init_func and rag_query_func.
    """
    # Optionally extend BASE_ARG_CONFIG for script-specific arguments
    arg_config = BASE_ARG_CONFIG.copy()
    arg_config["args"] += [
        {"flags": ["--retrieve_topk"],
         "params": {"type": int, "default": 5, "help": "Number of top documents to retrieve"}}
    ]
    arg_config["args"][0]["params"]["default"] = "LightRAG"
    args = parse_args(arg_config)

    # Initialize data
    corpus_data, grouped_questions = init_data(args)
    logging.info(f"🚀 Starting {args.method} processing for subset: {args.subset}")
    for item in corpus_data:
        corpus_name = item["corpus_name"]
        context = item["context"]
        asyncio.run(
            process_corpus(
                corpus_name=corpus_name,
                context=context,
                base_dir=args.base_dir,
                questions=grouped_questions,
                sample=args.sample,
                output_dir=f"./results/{args.method.lower()}",
                rag_init_func=lightrag_init_func,
                query_func=lightrag_query_func,
                args=args,
            )
        )

if __name__ == "__main__":
    main()