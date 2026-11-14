#!/bin/bash

# No RAG
python3 src/eval.py --log bpic20_international_declarations.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct --rag False
python3 src/eval.py --log bpic20_international_declarations.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id microsoft/phi-4 --rag False
python3 src/eval.py --log bpic20_international_declarations.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 32768 --rag False

python3 src/eval.py --log synthetic-13-multivariant.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct --rag False
python3 src/eval.py --log synthetic-13-multivariant.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id microsoft/phi-4 --rag False
python3 src/eval.py --log synthetic-13-multivariant.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 32768 --rag False

# Bucketing
python3 src/eval.py --log bpic20_international_declarations.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct --prefix_gap 3
python3 src/eval.py --log bpic20_international_declarations.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id microsoft/phi-4 --prefix_gap 3
python3 src/eval.py --log bpic20_international_declarations.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 32768 --prefix_gap 3

python3 src/eval.py --log synthetic-13-multivariant.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct --prefix_gap 3
python3 src/eval.py --log synthetic-13-multivariant.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id microsoft/phi-4 --prefix_gap 3
python3 src/eval.py --log synthetic-13-multivariant.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 32768 --prefix_gap 3

python3 src/eval.py --log bpic20_international_declarations.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct --prefix_gap 5
python3 src/eval.py --log bpic20_international_declarations.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id microsoft/phi-4 --prefix_gap 5
python3 src/eval.py --log bpic20_international_declarations.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 32768 --prefix_gap 5

python3 src/eval.py --log synthetic-13-multivariant.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct --prefix_gap 5
python3 src/eval.py --log synthetic-13-multivariant.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id microsoft/phi-4 --prefix_gap 5
python3 src/eval.py --log synthetic-13-multivariant.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 32768 --prefix_gap 5

# Embedding Models
python3 src/eval.py --log bpic20_international_declarations.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct --embed_model_id sentence-transformers/all-MiniLM-L6-v2
python3 src/eval.py --log bpic20_international_declarations.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id microsoft/phi-4 --embed_model_id sentence-transformers/all-MiniLM-L6-v2
python3 src/eval.py --log bpic20_international_declarations.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 32768 --embed_model_id sentence-transformers/all-MiniLM-L6-v2

python3 src/eval.py --log synthetic-13-multivariant.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct --embed_model_id sentence-transformers/all-MiniLM-L6-v2
python3 src/eval.py --log synthetic-13-multivariant.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id microsoft/phi-4 --embed_model_id sentence-transformers/all-MiniLM-L6-v2
python3 src/eval.py --log synthetic-13-multivariant.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 32768 --embed_model_id sentence-transformers/all-MiniLM-L6-v2

python3 src/eval.py --log bpic20_international_declarations.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct --embed_model_id sentence-transformers/all-mpnet-base-v2
python3 src/eval.py --log bpic20_international_declarations.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id microsoft/phi-4 --embed_model_id sentence-transformers/all-mpnet-base-v2
python3 src/eval.py --log bpic20_international_declarations.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 32768 --embed_model_id sentence-transformers/all-mpnet-base-v2

python3 src/eval.py --log synthetic-13-multivariant.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct --embed_model_id sentence-transformers/all-mpnet-base-v2
python3 src/eval.py --log synthetic-13-multivariant.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id microsoft/phi-4 --embed_model_id sentence-transformers/all-mpnet-base-v2
python3 src/eval.py --log synthetic-13-multivariant.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 32768 --embed_model_id sentence-transformers/all-mpnet-base-v2

python3 src/eval.py --log bpic20_international_declarations.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct --embed_model_id jinaai/jina-embeddings-v2-base-en
python3 src/eval.py --log bpic20_international_declarations.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id microsoft/phi-4 --embed_model_id jinaai/jina-embeddings-v2-base-en
python3 src/eval.py --log bpic20_international_declarations.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 32768 --embed_model_id jinaai/jina-embeddings-v2-base-en

python3 src/eval.py --log synthetic-13-multivariant.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct --embed_model_id jinaai/jina-embeddings-v2-base-en
python3 src/eval.py --log synthetic-13-multivariant.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id microsoft/phi-4 --embed_model_id jinaai/jina-embeddings-v2-base-en
python3 src/eval.py --log synthetic-13-multivariant.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 32768 --embed_model_id jinaai/jina-embeddings-v2-base-en

python3 src/eval.py --log bpic20_international_declarations.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct --embed_model_id jinaai/jina-embeddings-v3
python3 src/eval.py --log bpic20_international_declarations.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id microsoft/phi-4 --embed_model_id jinaai/jina-embeddings-v3
python3 src/eval.py --log bpic20_international_declarations.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 32768 --embed_model_id jinaai/jina-embeddings-v3

python3 src/eval.py --log synthetic-13-multivariant.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct --embed_model_id jinaai/jina-embeddings-v3
python3 src/eval.py --log synthetic-13-multivariant.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id microsoft/phi-4 --embed_model_id jinaai/jina-embeddings-v3
python3 src/eval.py --log synthetic-13-multivariant.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 32768 --embed_model_id jinaai/jina-embeddings-v3

python3 src/eval.py --log bpic20_international_declarations.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct --embed_model_id jinaai/jina-embeddings-v4
python3 src/eval.py --log bpic20_international_declarations.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id microsoft/phi-4 --embed_model_id jinaai/jina-embeddings-v4
python3 src/eval.py --log bpic20_international_declarations.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 32768 --embed_model_id jinaai/jina-embeddings-v4

python3 src/eval.py --log synthetic-13-multivariant.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct --embed_model_id jinaai/jina-embeddings-v4
python3 src/eval.py --log synthetic-13-multivariant.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id microsoft/phi-4 --embed_model_id jinaai/jina-embeddings-v4
python3 src/eval.py --log synthetic-13-multivariant.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 32768 --embed_model_id jinaai/jina-embeddings-v4

# Real-world logs
python3 src/eval.py --log sepsis.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Llama-3.1-8B-Instruct
python3 src/eval.py --log sepsis.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Llama-3.2-1B-Instruct
python3 src/eval.py --log sepsis.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Llama-3.2-3B-Instruct
python3 src/eval.py --log sepsis.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id mistralai/Mistral-7B-Instruct-v0.2
python3 src/eval.py --log sepsis.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id mistralai/Mistral-7B-Instruct-v0.3
python3 src/eval.py --log sepsis.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct
python3 src/eval.py --log sepsis.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id microsoft/phi-4
python3 src/eval.py --log sepsis.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id gpt-4o-mini
python3 src/eval.py --log sepsis.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 32768
python3 src/eval.py --log sepsis.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id deepseek-ai/DeepSeek-R1-Distill-Llama-8B --max_new_tokens 32768

python3 src/eval.py --log bpic20_international_declarations.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Llama-3.1-8B-Instruct
python3 src/eval.py --log bpic20_international_declarations.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Llama-3.2-1B-Instruct
python3 src/eval.py --log bpic20_international_declarations.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Llama-3.2-3B-Instruct
python3 src/eval.py --log bpic20_international_declarations.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id mistralai/Mistral-7B-Instruct-v0.2
python3 src/eval.py --log bpic20_international_declarations.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id mistralai/Mistral-7B-Instruct-v0.3
python3 src/eval.py --log bpic20_international_declarations.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct
python3 src/eval.py --log bpic20_international_declarations.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id microsoft/phi-4
python3 src/eval.py --log bpic20_international_declarations.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id gpt-4o-mini
python3 src/eval.py --log bpic20_international_declarations.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 32768
python3 src/eval.py --log bpic20_international_declarations.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id deepseek-ai/DeepSeek-R1-Distill-Llama-8B --max_new_tokens 32768

python3 src/eval.py --log hospital_billing.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Llama-3.1-8B-Instruct
python3 src/eval.py --log hospital_billing.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Llama-3.2-1B-Instruct
python3 src/eval.py --log hospital_billing.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Llama-3.2-3B-Instruct
python3 src/eval.py --log hospital_billing.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id mistralai/Mistral-7B-Instruct-v0.2
python3 src/eval.py --log hospital_billing.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id mistralai/Mistral-7B-Instruct-v0.3
python3 src/eval.py --log hospital_billing.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct
python3 src/eval.py --log hospital_billing.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id microsoft/phi-4
python3 src/eval.py --log hospital_billing.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id gpt-4o-mini
python3 src/eval.py --log hospital_billing.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 32768
python3 src/eval.py --log hospital_billing.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id deepseek-ai/DeepSeek-R1-Distill-Llama-8B --max_new_tokens 32768

python3 src/eval.py --log BPIC_2013_closed_problems.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Llama-3.1-8B-Instruct
python3 src/eval.py --log BPIC_2013_closed_problems.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Llama-3.2-1B-Instruct
python3 src/eval.py --log BPIC_2013_closed_problems.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Llama-3.2-3B-Instruct
python3 src/eval.py --log BPIC_2013_closed_problems.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id mistralai/Mistral-7B-Instruct-v0.2
python3 src/eval.py --log BPIC_2013_closed_problems.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id mistralai/Mistral-7B-Instruct-v0.3
python3 src/eval.py --log BPIC_2013_closed_problems.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct
python3 src/eval.py --log BPIC_2013_closed_problems.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id microsoft/phi-4
python3 src/eval.py --log BPIC_2013_closed_problems.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id gpt-4o-mini
python3 src/eval.py --log BPIC_2013_closed_problems.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 32768
python3 src/eval.py --log BPIC_2013_closed_problems.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id deepseek-ai/DeepSeek-R1-Distill-Llama-8B --max_new_tokens 32768

python3 src/eval.py --log BPIC_2013_incidents.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Llama-3.1-8B-Instruct
python3 src/eval.py --log BPIC_2013_incidents.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Llama-3.2-1B-Instruct
python3 src/eval.py --log BPIC_2013_incidents.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Llama-3.2-3B-Instruct
python3 src/eval.py --log BPIC_2013_incidents.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id mistralai/Mistral-7B-Instruct-v0.2
python3 src/eval.py --log BPIC_2013_incidents.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id mistralai/Mistral-7B-Instruct-v0.3
python3 src/eval.py --log BPIC_2013_incidents.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct
python3 src/eval.py --log BPIC_2013_incidents.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id microsoft/phi-4
python3 src/eval.py --log BPIC_2013_incidents.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id gpt-4o-mini
python3 src/eval.py --log BPIC_2013_incidents.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 32768
python3 src/eval.py --log BPIC_2013_incidents.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id deepseek-ai/DeepSeek-R1-Distill-Llama-8B --max_new_tokens 32768

python3 src/eval.py --log BPIC_2012.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Llama-3.1-8B-Instruct
python3 src/eval.py --log BPIC_2012.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Llama-3.2-1B-Instruct
python3 src/eval.py --log BPIC_2012.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Llama-3.2-3B-Instruct
python3 src/eval.py --log BPIC_2012.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id mistralai/Mistral-7B-Instruct-v0.2
python3 src/eval.py --log BPIC_2012.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id mistralai/Mistral-7B-Instruct-v0.3
python3 src/eval.py --log BPIC_2012.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct
python3 src/eval.py --log BPIC_2012.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id microsoft/phi-4
python3 src/eval.py --log BPIC_2012.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id gpt-4o-mini
python3 src/eval.py --log BPIC_2012.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 32768
python3 src/eval.py --log BPIC_2012.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id deepseek-ai/DeepSeek-R1-Distill-Llama-8B --max_new_tokens 32768


# Synthetic logs
python3 src/eval.py --log udonya.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Llama-3.1-8B-Instruct
python3 src/eval.py --log udonya.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Llama-3.2-1B-Instruct
python3 src/eval.py --log udonya.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Llama-3.2-3B-Instruct
python3 src/eval.py --log udonya.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id mistralai/Mistral-7B-Instruct-v0.2
python3 src/eval.py --log udonya.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id mistralai/Mistral-7B-Instruct-v0.3
python3 src/eval.py --log udonya.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct
python3 src/eval.py --log udonya.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id microsoft/phi-4
python3 src/eval.py --log udonya.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id gpt-4o-mini
python3 src/eval.py --log udonya.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 32768
python3 src/eval.py --log udonya.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id deepseek-ai/DeepSeek-R1-Distill-Llama-8B --max_new_tokens 32768

python3 src/eval.py --log synthetic-13-multivariant.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Llama-3.1-8B-Instruct
python3 src/eval.py --log synthetic-13-multivariant.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Llama-3.2-1B-Instruct
python3 src/eval.py --log synthetic-13-multivariant.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Llama-3.2-3B-Instruct
python3 src/eval.py --log synthetic-13-multivariant.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id mistralai/Mistral-7B-Instruct-v0.2
python3 src/eval.py --log synthetic-13-multivariant.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id mistralai/Mistral-7B-Instruct-v0.3
python3 src/eval.py --log synthetic-13-multivariant.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct
python3 src/eval.py --log synthetic-13-multivariant.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id microsoft/phi-4
python3 src/eval.py --log synthetic-13-multivariant.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id gpt-4o-mini
python3 src/eval.py --log synthetic-13-multivariant.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 32768
python3 src/eval.py --log synthetic-13-multivariant.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id deepseek-ai/DeepSeek-R1-Distill-Llama-8B --max_new_tokens 32768

python3 src/eval.py --log synthetic-2-2var-1rel-1-nonrel.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Llama-3.1-8B-Instruct
python3 src/eval.py --log synthetic-2-2var-1rel-1-nonrel.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Llama-3.2-1B-Instruct
python3 src/eval.py --log synthetic-2-2var-1rel-1-nonrel.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Llama-3.2-3B-Instruct
python3 src/eval.py --log synthetic-2-2var-1rel-1-nonrel.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id mistralai/Mistral-7B-Instruct-v0.2
python3 src/eval.py --log synthetic-2-2var-1rel-1-nonrel.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id mistralai/Mistral-7B-Instruct-v0.3
python3 src/eval.py --log synthetic-2-2var-1rel-1-nonrel.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct
python3 src/eval.py --log synthetic-2-2var-1rel-1-nonrel.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id microsoft/phi-4
python3 src/eval.py --log synthetic-2-2var-1rel-1-nonrel.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id gpt-4o-mini
python3 src/eval.py --log synthetic-2-2var-1rel-1-nonrel.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 32768
python3 src/eval.py --log synthetic-2-2var-1rel-1-nonrel.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id deepseek-ai/DeepSeek-R1-Distill-Llama-8B --max_new_tokens 32768

# Fine-tuned model
# Real-world logs
python3 src/eval.py --log sepsis.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id angeloc1/autotrain-phi-ft-nap2
python3 src/eval.py --log bpic20_international_declarations.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id angeloc1/autotrain-phi-ft-nap2
python3 src/eval.py --log hospital_billing.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id angeloc1/autotrain-phi-ft-nap2
python3 src/eval.py --log BPIC_2013_closed_problems.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id angeloc1/autotrain-phi-ft-nap2
python3 src/eval.py --log BPIC_2013_incidents.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id angeloc1/autotrain-phi-ft-nap2
python3 src/eval.py --log BPIC_2012.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id angeloc1/autotrain-phi-ft-nap2
# Synthetic logs
python3 src/eval.py --log udonya.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id angeloc1/autotrain-phi-ft-nap2
python3 src/eval.py --log synthetic-13-multivariant.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id angeloc1/autotrain-phi-ft-nap2
python3 src/eval.py --log synthetic-2-2var-1rel-1-nonrel.xes --evaluation_modality evaluation-attributes --rebuild_db_and_tests True --llm_id angeloc1/autotrain-phi-ft-nap2