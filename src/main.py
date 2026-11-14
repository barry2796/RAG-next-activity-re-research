from argparse import ArgumentParser
from dotenv import load_dotenv
import os
import torch
import warnings

import log_preprocessing as lp
import pipeline as p
import utility as u
import vector_store as vs

DEVICE = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
load_dotenv()
HF_AUTH = os.getenv('HF_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
URL = os.getenv('QDRANT_URL')
GRPC_PORT = int(os.getenv('QDRANT_GRPC_PORT'))
COLLECTION_NAME = 'nap-rag'
SEED = 10
warnings.filterwarnings('ignore')


def generate_live_response(pipeline, question: str, curr_datetime: str, vectordb, num_chunks: int, info_run: dict):
    complete_prompt, answer = pipeline.produce_answer(question, vectordb, num_chunks, info_run)
    print(f'Prompt: {complete_prompt}\n')
    print(f'Answer: {answer}\n')
    print('--------------------------------------------------')

    u.log_to_file(f'Query: {complete_prompt}\n\nAnswer: {answer}\n\n##########################\n\n',
                  curr_datetime, info_run)


def live_prompting(pipeline, vect_db, num_chunks: int, info_run: dict):
    import datetime
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    while True:
        query = input('Insert the query (type "quit" to exit): ')

        if query.lower() == 'quit':
            print("Exiting the chat.")
            break

        generate_live_response(pipeline, query, current_datetime, vect_db, num_chunks, info_run)
        print()


def parse_arguments():
    parser = ArgumentParser(description="Run LLM Generation in Live Mode.")
    parser.add_argument('--embed_model_id', type=str, default='sentence-transformers/all-MiniLM-L12-v2',
                        help='Embedding model identifier')
    parser.add_argument('--vector_dimension', type=int, default=384,
                        help='Vector space dimension')
    parser.add_argument('--llm_id', type=str, default='gpt-4.1',
                        help='LLM model identifier')
    parser.add_argument('--model_max_length', type=int, help='Maximum input length (context window)',
                        default=128000)
    parser.add_argument('--num_documents_in_context', type=int, help='Number of documents in the context',
                        default=3)
    parser.add_argument('--log', type=str, help='The event log to use for the next activity prediction',
                        default='sepsis.xes')
    parser.add_argument('--prefix_base', type=int, help='Base number of events in a prefix trace',
                        default=1)
    parser.add_argument('--prefix_gap', type=int, help='Gap number of events in a prefix trace',
                        default=3)
    parser.add_argument('--max_new_tokens', type=int, help='Maximum number of tokens to generate',
                        default=2048)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--rebuild_db_and_tests', type=u.str2bool,
                        help='Rebuild the vector index and the test set', default=True)
    parser.add_argument('--rag', type=u.str2bool,
                        help='Support for Retrieval-Augmented Generation', default=True)
    parser.add_argument('--evaluation_modality', type=str, default='evaluation-attributes',
                        help='Evaluation modality to use (e.g., evaluation-concept_names, evaluation-attributes)')
    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()
    embed_model_id = args.embed_model_id
    embed_model, actual_dimension = p.initialize_embedding_model(embed_model_id, DEVICE, args.batch_size)
    
    # Use the actual dimension from the model, not the command line argument
    space_dimension = actual_dimension
    print(f"Using embedding dimension: {space_dimension}")
    
    rag = args.rag
    q_client, q_store = vs.initialize_vector_store(URL, GRPC_PORT, COLLECTION_NAME, embed_model, space_dimension, args.rebuild_db_and_tests)
    num_docs = args.num_documents_in_context
    event_attributes = []
    activities_set = set()
    total_traces_size = 0
    traces_to_store_size = 0
    
    if args.rebuild_db_and_tests:
        content = lp.read_event_log(args.log)
        traces = lp.extract_traces(content)
        prefixes = lp.build_prefixes(traces)
        prefixes, event_attributes, activities_set = lp.process_prefixes(prefixes, args.log)
        traces_to_store_size = len(prefixes)
        vs.store_traces(prefixes, q_client, args.log, embed_model, COLLECTION_NAME)

    model_id = args.llm_id
    max_new_tokens = args.max_new_tokens
    os.environ['HF_TOKEN'] = HF_AUTH if HF_AUTH else ""
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY if OPENAI_API_KEY else ""
    pipeline = p.RAGPipeline(model_id, max_new_tokens, rag)
    run_data = {
        'Batch Size': args.batch_size,
        'Embedding Model ID': embed_model_id,
        'Vector Space Dimension': space_dimension,
        'Evaluation Modality': args.evaluation_modality,
        'Event Log': args.log,
        'Total Traces in Log': total_traces_size,
        'Traces Stored Size': traces_to_store_size,
        'LLM ID': model_id,
        'Context Window LLM': args.model_max_length,
        'Max Generated Tokens LLM': max_new_tokens,
        'Number of Documents in the Context': num_docs,
        'Prefix Base': args.prefix_base,
        'Prefix Gap': args.prefix_gap,
        'Rebuilt Vector Index and Test Set': args.rebuild_db_and_tests,
        'RAG': rag
    }
    if event_attributes:
        run_data['Event Attributes'] = str(event_attributes)
        run_data['Activities'] = activities_set

    print("Starting live prompting mode...")
    live_prompting(pipeline, q_store, num_docs, run_data)


if __name__ == "__main__":
    u.seed_everything(SEED)
    main()
