import asyncio
import logging
import os

from fast_graphrag import GraphRAG
from fast_graphrag._llm import OpenAILLMService, HuggingFaceEmbeddingService
from transformers import AutoTokenizer, AutoModel

from Evaluation.llm.ollama_client import OllamaClient, OllamaWrapper
from Examples.shared_code import BASE_ARG_CONFIG, parse_args, init_data, process_corpus, setup_logging

logger = setup_logging("fast_graphrag_processing.log")

# Configuration constants
DOMAIN = "Analyze this story and identify the characters. Focus on how they interact with each other, the locations they explore, and their relationships."
EXAMPLE_QUERIES = [
    "What is the significance of Christmas Eve in A Christmas Carol?",
    "How does the setting of Victorian London contribute to the story's themes?",
    "Describe the chain of events that leads to Scrooge's transformation.",
    "How does Dickens use the different spirits (Past, Present, and Future) to guide Scrooge?",
    "Why does Dickens choose to divide the story into \"staves\" rather than chapters?"
]
ENTITY_TYPES = ["Character", "Animal", "Place", "Object", "Activity", "Event"]

async def fastgrag_init_func(
    corpus_name: str,
    context: str,
    base_dir: str,
    args) -> GraphRAG:
    """Initialize GraphRAG instance for a specific corpus"""
    try:
        embedding_tokenizer = AutoTokenizer.from_pretrained(args.embed_model)
        embedding_model = AutoModel.from_pretrained(args.embed_model)
        logging.info(f"✅ Loaded embedding model: {args.embed_model}")
    except Exception as e:
        logging.error(f"❌ Failed to load embedding model: {e}")
        return

    if args.mode == "ollama":
        # Create Ollama client
        ollama_client = OllamaClient(base_url=args.llm_base_url)
        llm_service = OllamaWrapper(ollama_client, args.model_name)
        logging.info(f"✅ Using Ollama LLM service: {args.model_name} at {args.llm_base_url}")
    else:
        # Use OpenAI-compatible service
        llm_service = OpenAILLMService(
            model=args.model_name,
            base_url=args.llm_base_url,
            api_key=args.llm_api_key,
        )
        logging.info(f"✅ Using OpenAI-compatible LLM service: {args.model_name} at {args.llm_base_url}")

    # Initialize Fast-GraphRAG
    grag = GraphRAG(
        working_dir=os.path.join(base_dir, corpus_name),
        domain=DOMAIN,
        example_queries="\n".join(EXAMPLE_QUERIES),
        entity_types=ENTITY_TYPES,
        config=GraphRAG.Config(
            llm_service=llm_service,
            embedding_service=HuggingFaceEmbeddingService(
                model=embedding_model,
                tokenizer=embedding_tokenizer,
                embedding_dim=1024,
                max_token_size=8192
            ),
        ),
    )
    return grag

async def fastgrag_query_func(
    grag: GraphRAG,
    question: str,
    args
):
    """Query the GraphRAG instance"""
    # Execute query
    response = grag.query(question["question"])
    context_chunks = response.to_dict()['context']['chunks']
    context = [item[0]["content"] for item in context_chunks]
    predicted_answer = response.response
    return predicted_answer, context


def main():
    """
        Main function to process corpora and answer questions using Method (Fast-GraphRAG).
        This can be used as a template for other (Graph)RAG systems.
        The main function should not require changes beyond
        - adding commandline arguments and
        - changing the rag_init_func and rag_query_func.
    """
    # Optionally extend BASE_ARG_CONFIG for script-specific arguments
    arg_config = BASE_ARG_CONFIG.copy()
    arg_config["args"][0]["params"]["default"] = "FastGraphRAG"
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
                rag_init_func=fastgrag_init_func,
                query_func=fastgrag_query_func,
                args=args,
            )
        )

if __name__ == "__main__":
    main()