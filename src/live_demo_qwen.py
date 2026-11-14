import os
import sys
import json
import random
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
import torch

load_dotenv()

# An empty-string HF/OpenAI key (as left in .env) breaks public model downloads / client init.
if not os.getenv('HF_TOKEN'):
    os.environ.pop('HF_TOKEN', None)
if not os.getenv('OPENAI_API_KEY'):
    os.environ.pop('OPENAI_API_KEY', None)

import log_preprocessing as lp
import pipeline as p
import utility as u
import vector_store as vs

DEVICE = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
URL = os.getenv('QDRANT_URL')
GRPC_PORT = int(os.getenv('QDRANT_GRPC_PORT'))
COLLECTION_NAME = 'nap-rag'
SEED = 10
u.seed_everything(SEED)

LOG_FILE = 'sepsis.xes'
MODEL_ID = 'Qwen/Qwen2.5-3B-Instruct'
MAX_NEW_TOKENS = 1536
NUM_DOCS = 3
N_SAMPLES = 5

print('Step 1/5: Loading embedding model...', flush=True)
embed_model, dim = p.initialize_embedding_model('sentence-transformers/all-MiniLM-L12-v2', DEVICE, 8)

print('Step 2/5: Preparing Qdrant vector store (rebuild)...', flush=True)
q_client, q_store = vs.initialize_vector_store(URL, GRPC_PORT, COLLECTION_NAME, embed_model, dim, True)

print('Step 3/5: Preprocessing sepsis.xes event log...', flush=True)
content = lp.read_event_log(LOG_FILE)
traces = lp.extract_traces(content)
prefixes_raw = lp.build_prefixes(traces)
prefixes, event_attributes, activities_set = lp.process_prefixes(prefixes_raw, LOG_FILE)
vs.store_traces(prefixes, q_client, LOG_FILE, embed_model, COLLECTION_NAME)

items = list(prefixes.items())
random.shuffle(items)
samples = items[:N_SAMPLES]
demo_questions = []
for prefix_str, prediction_str in samples:
    attrs = prediction_str.split('|')[0].replace('Values:', '').strip()
    next_activity = prediction_str.split('| Next activity: ')[1].strip()
    question = f'{prefix_str} - Values: {attrs}'
    demo_questions.append((question, next_activity))

print(f'Step 4/5: Loading LLM {MODEL_ID} (4-bit quantized)...', flush=True)
pipeline_instance = p.RAGPipeline(MODEL_ID, MAX_NEW_TOKENS, use_rag=True)

run_data = {
    'Batch Size': 8,
    'Embedding Model ID': 'sentence-transformers/all-MiniLM-L12-v2',
    'Vector Space Dimension': dim,
    'Evaluation Modality': 'evaluation-attributes',
    'Event Log': LOG_FILE,
    'LLM ID': MODEL_ID,
    'Max Generated Tokens LLM': MAX_NEW_TOKENS,
    'Number of Documents in the Context': NUM_DOCS,
    'RAG': True,
    'Event Attributes': str(event_attributes),
    'Activities': activities_set,
}

print('Step 5/5: Running live demo predictions...', flush=True)
results = []
curr_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
for i, (question, expected) in enumerate(demo_questions, 1):
    print(f'--- Query {i}/{len(demo_questions)} ---', flush=True)
    prompt, answer = pipeline_instance.produce_answer(question, q_store, NUM_DOCS, run_data)
    print(f'Prefix: {question}', flush=True)
    print(f'Expected next activity: {expected}', flush=True)
    print(f'Model answer: {answer}', flush=True)
    u.log_to_file(
        f'Query: {prompt}\n\nAnswer: {answer}\n\nExpected: {expected}\n\n##########################\n\n',
        curr_datetime, run_data
    )
    results.append({
        'index': i,
        'prefix': question,
        'expected_next_activity': expected,
        'model_answer': answer,
    })

out_json = os.path.join(os.path.dirname(__file__), '..', 'tests', 'outputs', f'live_demo_results_{curr_datetime}.json')
with open(out_json, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print('DONE', flush=True)
print(f'Text log: tests/outputs/output_{curr_datetime}.txt', flush=True)
print(f'JSON results: {out_json}', flush=True)
