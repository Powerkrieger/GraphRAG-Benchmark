import asyncio
import logging
import os
from typing import List

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import HippoRAG components after setting environment
from transformers import AutoTokenizer
from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.utils.misc_utils import string_to_bool
from src.hipporag.utils.config_utils import BaseConfig

# import local code
from Examples.shared_code import BASE_ARG_CONFIG, parse_args, setup_logging, init_data, process_corpus

logger = setup_logging("hipporag_processing.log")

def split_text(
    text: str, 
    tokenizer: AutoTokenizer, 
    chunk_token_size: int = 256, 
    chunk_overlap_token_size: int = 32
) -> List[str]:
    """Split text into chunks based on token length with overlap"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    start = 0
    while start < len(tokens):
        end = min(start + chunk_token_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(
            chunk_tokens, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )
        chunks.append(chunk_text)
        if end == len(tokens):
            break
        start += chunk_token_size - chunk_overlap_token_size
    return chunks

async def hipporag_init_func(corpus_name, context, base_dir, args):
    tokenizer = AutoTokenizer.from_pretrained(args.embed_model_path)
    chunks = split_text(context, tokenizer)
    docs = [f'{idx}:{chunk}' for idx, chunk in enumerate(chunks)]
    config = BaseConfig(
        save_dir=os.path.join(base_dir, corpus_name),
        llm_base_url=args.llm_base_url,
        llm_name=args.model_name,
        embedding_model_name=args.embed_model,
        force_index_from_scratch=True,
        force_openie_from_scratch=True,
        rerank_dspy_file_path="src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=5,
        linking_top_k=5,
        max_qa_steps=3,
        qa_top_k=5,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=8,
        max_new_tokens=None,
        corpus_len=len(docs),
        openie_mode="online"
    )

    # Override LLM configuration for Ollama mode
    if args.mode == "ollama":
        config.llm_mode = "ollama"
        logging.info(f"✅ Using Ollama mode: {args.model_name} at {args.llm_base_url}")
    else:
        config.llm_mode = "openai"
        logging.info(f"✅ Using OpenAI mode: {args.model_name} at {args.llm_base_url}")

    # Initialize HippoRAG
    hipporag = HippoRAG(global_config=config)
    hipporag.index(docs)
    return hipporag

async def hipporag_query_func(rag, question, args) -> (str, list):
    """
    Query HippoRAG with a question and return the response and context
    Take:
    - rag: Initialized HippoRAG instance
    - question: Dictionary with 'question' key
    - args: Parsed command-line arguments
    Return:
    - response: Generated answer string
    - context: List of context documents used for the answer
    """
    queries = [question["question"]]
    gold_answers = [[question['answer']]]
    queries_solutions, _, _, _, _ = rag.rag_qa(queries=queries, gold_docs=None, gold_answers=gold_answers)
    solution = queries_solutions[0].to_dict()
    return solution.get("answer", ""), solution.get("docs", "")


def main():
    """
        Main function to process corpora and answer questions using Method (HippoRag).
        This can be used as a template for other (Graph)RAG systems.
        The main function should not require changes beyond
        - adding commandline arguments and
        - changing the rag_init_func and rag_query_func.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Optionally extend BASE_ARG_CONFIG for script-specific arguments
    arg_config = BASE_ARG_CONFIG.copy()
    arg_config["args"][0]["params"]["default"] = "HippoRAG"
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
                output_dir=f"./results/{args.method.tolower()}",
                rag_init_func=hipporag_init_func,
                query_func=hipporag_query_func,
                args=args,
            )
        )

if __name__ == "__main__":
    main()