import asyncio
import logging
import os
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from Core.GraphRAG import GraphRAG
from Core.Utils.Evaluation import Evaluator
from Data.QueryDataset import RAGQueryDataset
from Option.Config2 import Config

from shared_code import setup_logging, BASE_ARG_CONFIG, parse_args, init_data, process_corpus

logger = setup_logging("digimon_processing.log")

async def initialize_rag(
    config_path: Path,
    source: str,
    args,
) -> GraphRAG:
    """Initialize GraphRAG instance for a specific source"""
    logger.info(f"🛠️ Initializing GraphRAG for source: {source}")
    
    # TODO: Add support for ollama
    if args.mode == "ollama":
        # For Ollama mode, we need to create a custom config
        # This is a simplified approach - you may need to adjust based on your Config class
        opt = Config.parse(config_path, dataset_name=source)

        # Override LLM settings for Ollama
        if hasattr(opt, 'llm_config'):
            opt.llm_config.model_name = args.model_name
            opt.llm_config.base_url = args.llm_base_url
            opt.llm_config.api_key = args.llm_api_key
            opt.llm_config.mode = args.mode

        logger.info(f"Ollama configuration: model={args.model_name}, base_url={args.llm_base_url}")
    else:
        # Parse configuration normally
        opt = Config.parse(config_path, dataset_name=source)
        logger.info(f"Configuration parsed: {opt}")
    
    # Create RAG instance
    rag = GraphRAG(config=opt)
    logger.info(f"✅ GraphRAG initialized for {source}")
    return rag

async def digimon_init_func(corpus_name, context, base_dir, args):
    """Initialize Digimon (GraphRAG) instance for a specific corpus"""
    rag = await initialize_rag(
        config_path=Path(args.config),
        source=corpus_name,
        args=args,
    )
    # Index the corpus
    corpus = [{
        "title": corpus_name,
        "content": context,
        "doc_id": 0,
    }]
    await rag.insert(corpus)
    return rag

async def digimon_query_func(
    rag: GraphRAG,
    question: dict,
    args
):
    """Query the GraphRAG instance"""
    predicted_answer, context = await rag.query(question["question"])
    return predicted_answer, context

def main():
    """
        Main function to process corpora and answer questions using Method (Digimon).
        This can be used as a template for other (Graph)RAG systems.
        The main function should not require changes beyond
        - adding commandline arguments and
        - changing the rag_init_func and rag_query_func.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Optionally extend BASE_ARG_CONFIG for script-specific arguments
    arg_config = BASE_ARG_CONFIG.copy()
    arg_config["args"] += [
        {"flags": ["--config"],
         "params": {"default": "./config.yml", "help": "Path to configuration YAML file"}}
    ]
    arg_config["args"][0]["params"]["default"] = "Digimon"
    args = parse_args(arg_config)

    # Initialize data
    corpus_data, grouped_questions = init_data(args)
    logging.info(f"🚀 Starting {args.method} processing for subset: {args.subset}")
    for corpus_name, context in corpus_data.items():
        asyncio.run(
            process_corpus(
                corpus_name=corpus_name,
                context=context,
                base_dir=args.base_dir,
                questions=grouped_questions,
                sample=args.sample,
                output_dir=f"./results/{args.method.lower()}",
                rag_init_func=digimon_init_func,
                query_func=digimon_query_func,
                args=args,
            )
        )

if __name__ == "__main__":
    main()