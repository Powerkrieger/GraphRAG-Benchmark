import os
import json
import logging
from typing import Dict, List

from tqdm import tqdm


def setup_logging(logfile: str = None, level: int = logging.INFO):
    handlers = [logging.StreamHandler()]
    if logfile:
        handlers.append(logging.FileHandler(logfile))
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=level,
        handlers=handlers
    )
    return logging.getLogger(__name__)


import argparse

# Define subset paths
SUBSET_PATHS = {
    "medical": {
        "corpus": "./Datasets/Corpus/medical.json",
        "questions": "./Datasets/Questions/medical_questions.json"
    },
    "novel": {
        "corpus": "./Datasets/Corpus/novel.json",
        "questions": "./Datasets/Questions/novel_questions.json"
    }
}


BASE_ARG_CONFIG = {
    "description": "Process Corpora and Answer Questions",
    "args": [
        {"flags": ["--method"], "params": {"required": True, "type":str, "help": "Method name for logging and folder naming"}},
        {"flags": ["--subset"], "params": {"required": True, "choices": ["medical", "novel"], "help": "Subset to process"}},
        {"flags": ["--base_dir"], "params": {"default": "./workspace", "help": "Base working directory"}},
        {"flags": ["--mode"], "params": {"required": True, "choices": ["API", "ollama"], "help": "Use API or ollama for LLM"}},
        {"flags": ["--model_name"], "params": {"default": "gpt-4o-mini", "help": "LLM model identifier"}},
        {"flags": ["--embed_model"], "params": {"default": "bge-large:latest", "help": "Embedding model name"}},
        {"flags": ["--sample"], "params": {"type": int, "default": None, "help": "Number of questions to sample per corpus"}},
        {"flags": ["--llm_base_url"], "params": {"default": "https://api.openai.com/v1", "help": "Base URL for LLM API"}},
        {"flags": ["--llm_api_key"], "params": {"default": os.getenv("OPENAI_API_KEY", ""), "help": "API key for LLM service"}},
    ]
}

def parse_args(arg_config):
    parser = argparse.ArgumentParser(description=arg_config.get("description", "Process Corpora and Answer Questions"))
    for arg in arg_config["args"]:
        parser.add_argument(*arg["flags"], **arg["params"])
    args = parser.parse_args()
    # Validate subset
    if args.subset not in SUBSET_PATHS:
        logging.error(f"❌ Invalid subset: {args.subset}. Valid options: {list(SUBSET_PATHS.keys())}")
        return
    if args.mode not in ["API", "ollama"]:
        logging.error(f'Invalid mode: {args.subset}. Valid options: {["API", "ollama"]}')
        return
    # Handle API key security
    if not args.llm_api_key and args.mode == "API":
        logging.warning("⚠️ No API key provided! Requests may fail.")
    # Create workspace directory
    os.makedirs(args.base_dir, exist_ok=True)
    return args


def init_data(args):
    # Get file paths for this subset
    corpus_path = SUBSET_PATHS[args.subset]["corpus"]
    questions_path = SUBSET_PATHS[args.subset]["questions"]

    # Load corpus data
    try:
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus_data = json.load(f)
        logging.info(f"📖 Loaded corpus with {len(corpus_data)} documents from {corpus_path}")
    except Exception as e:
        logging.error(f"❌ Failed to load corpus: {e}")
        return

    # Sample corpus data if requested
    if args.sample:
        corpus_data = corpus_data[:1]

    # Load question data
    try:
        with open(questions_path, "r", encoding="utf-8") as f:
            question_data = json.load(f)
        grouped_questions = group_questions_by_source(question_data)
        logging.info(f"❓ Loaded questions with {len(question_data)} entries from {questions_path}")
    except Exception as e:
        logging.error(f"❌ Failed to load questions: {e}")
        return

    return corpus_data, grouped_questions


def group_questions_by_source(question_list: List[dict]) -> Dict[str, List[dict]]:
    grouped_questions = {}
    for question in question_list:
        source = question.get("source")
        if source not in grouped_questions:
            grouped_questions[source] = []
        grouped_questions[source].append(question)
    return grouped_questions


async def process_corpus(
        corpus_name: str,
        context: str,
        base_dir: str,
        questions: Dict[str, List[dict]],
        sample: int,
        output_dir: str,
        rag_init_func,
        query_func,
        args,
):
    """Process a single corpus: index it and answer its questions"""
    logging.info(f"📚 Processing corpus: {corpus_name}")

    # Initialize RAG and index corpus
    rag = await rag_init_func(corpus_name, context, base_dir, args)

    logging.info(f"✅ Indexed corpus: {corpus_name} ({len(context.split())} words)")

    # Prepare output path
    corpus_output_dir = os.path.join(output_dir, corpus_name)
    os.makedirs(corpus_output_dir, exist_ok=True)
    output_path = os.path.join(corpus_output_dir, f"predictions_{corpus_name}.json")

    # Get questions for this corpus
    corpus_questions = questions.get(corpus_name, [])
    if not corpus_questions:
        logging.warning(f"⚠️ No questions found for corpus: {corpus_name}")
        return

    # Sample questions if requested
    if sample and sample < len(corpus_questions):
        corpus_questions = corpus_questions[:sample]

    logging.info(f"🔍 Found {len(corpus_questions)} questions for {corpus_name}")

    # Process questions
    results = []
    for q in tqdm(corpus_questions, desc=f"Answering questions for {corpus_name}"):
        try:
            predicted_answer, context = await query_func(rag, q, args)
            # Collect results
            results.append({
                "id": q["id"],
                "question": q["question"],
                "source": corpus_name,
                "context": context,
                "evidence": q.get("evidence", ""),
                "question_type": q.get("question_type", ""),
                "generated_answer": predicted_answer,
                "ground_truth": q.get("answer", ""),
            })
        except Exception as e:
            logging.error(f"❌ Error processing question {q.get('id')}: {e}")
            results.append({
                "id": q["id"],
                "error": str(e)
            })

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logging.info(f"💾 Saved {len(results)} predictions to: {output_path}")
