import json
import os
from typing import Dict, List, Tuple

import torch
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, AutoConfig
from torch import bfloat16
from sentence_transformers import SentenceTransformer

from vector_store import retrieve_context


class RAGPipeline:
    MODELS = {
        'api': {
            'openai': ['gpt-4o-mini', 'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-4.1', 'gpt-4o'],
            'google_genai': ['gemini-2.0-flash', 'gemini-2.5-flash-preview-05-20', 'gemini-2.5-pro', 'gemini-2.5-flash'],
            'deepseek': ['deepseek-chat', 'deepseek-reasoner'],
            'anthropic': [],
        },
        'local': {
            'metaai': ['meta-llama/Meta-Llama-3-8B-Instruct', 'meta-llama/Llama-3.1-8B-Instruct',
                       'meta-llama/Llama-3.2-1B-Instruct', 'meta-llama/Llama-3.2-3B-Instruct'],
            'mistral': ['mistralai/Mistral-7B-Instruct-v0.2','mistralai/Mistral-7B-Instruct-v0.3',
                        'mistralai/Mistral-Nemo-Instruct-2407', 'mistralai/Ministral-8B-Instruct-2410'],
            'qwen': ['Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-3B-Instruct', 'Qwen/Qwen2.5-1.5B-Instruct'],
            'google_genai': ['google/gemma-2-9b-it'],
            'microsoft': ['microsoft/phi-4'],
            'deepseek': ['deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'],
            'angeloc1': ['angeloc1/autotrain-phi-ft-nap']
        }
    }
    TERMINATOR_TOKENS = {
        'metaai': "<|eot_id|>",
        'mistral': "[/INST]]",
        'qwen': "<|im_end|>",
        'microsoft': "<|im_sep|>"
    }
    TEMPLATE_MAPPING = {
        'metaai': 'template-llama_instruct',
        'mistral': 'template-mistral',
        'qwen': 'template-qwen',
        'microsoft': 'template-phi',
        'angeloc1': 'template-phi-ft',
        'deepseek': 'template-deepseek',
        'openai': 'template-generic',
        'google_genai': 'template-generic',
        'anthropic': 'template-generic'
    }
    RESPONSE_DELIMITERS = {
        'metaai': '<|start_header_id|>assistant<|end_header_id|>',
        'mistral': '[/INST]',
        'qwen': '<|im_start|>assistant',
        'microsoft': '<|im_start|>assistant<|im_sep|>',
        'angeloc1': '<|im_start|>assistant<|im_sep|>',
        'deepseek': 'Assistant: '
    }

    def __init__(self, model_id: str, max_new_tokens: int, use_rag: bool = True) -> None:
        self.model_id = model_id
        self.model_family, self.model_type = self._get_model_family_type()
        if not self.model_family:
            raise ValueError(f"Could not determine model family for ID: {self.model_id}. Please check MODELS definition.")
        self.max_new_tokens = max_new_tokens
        self.use_rag = use_rag
        self.llm = self._initialize_model()
        self.prompt_template = self._generate_prompt_template()
        self.chain = self.prompt_template | self.llm

    def _get_model_family_type(self) -> Tuple[str, str]:
        model_id_lower = self.model_id.lower()
        for model_type, families in self.MODELS.items():
            for family, models_in_family in families.items():
                if any(model_id_lower == model.lower() or model_id_lower in model.lower() for model in models_in_family):
                    return family, model_type
        
        print(f"Warning: Model ID '{self.model_id}' not found in predefined MODELS. Defaulting to 'openai', 'api'.")
        return "openai", "api"

    def _initialize_model(self):
        if self.model_type == 'local':
            HF_AUTH = os.getenv('HF_TOKEN')
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=bfloat16,
                llm_int8_enable_fp32_cpu_offload=True
            )
            model_config = AutoConfig.from_pretrained(self.model_id, token=HF_AUTH)

            max_memory = None
            if torch.cuda.is_available():
                total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                gpu_budget_gb = max(total_gb - 1.0, 1.0)  # leave ~1GiB headroom for activations/other processes
                max_memory = {0: f"{gpu_budget_gb:.1f}GiB", "cpu": "32GiB"}

            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                config=model_config,
                quantization_config=bnb_config,
                device_map='auto',
                max_memory=max_memory,
                token=HF_AUTH
            )
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=HF_AUTH)

            pipeline_params = {
                "model": model,
                "tokenizer": tokenizer,
                "return_full_text": True,
                "task": "text-generation",
                "do_sample": True,
                "temperature": 0.1,
                "max_new_tokens": self.max_new_tokens,
                "repetition_penalty": 1.1
            }

            model_family_key = self.model_family.lower()
            if model_family_key in self.TERMINATOR_TOKENS:
                terminators = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids(self.TERMINATOR_TOKENS[model_family_key])
                ]
                pipeline_params["eos_token_id"] = terminators
                pipeline_params["pad_token_id"] = tokenizer.eos_token_id
            
            generate_text = pipeline(**pipeline_params)
            return HuggingFacePipeline(pipeline=generate_text) 
            
        elif self.model_type == 'api':
            return init_chat_model(
                self.model_id,
                model_provider=self.model_family,
                temperature=0.1,
                max_tokens=self.max_new_tokens
            )
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}. Must be 'local' or 'api'.")

    def _generate_prompt_template(self) -> PromptTemplate:
        prompts_file = 'prompts.json' if self.use_rag else 'prompts_no_rag.json'
        path_prompts = os.path.join(os.path.dirname(__file__), prompts_file)
        
        with open(path_prompts, 'r', encoding='utf-8') as prompt_file:
            prompts = json.load(prompt_file)

        model_family_key = self.model_family.lower()
        template_key = self.TEMPLATE_MAPPING.get(model_family_key, 'template-generic')
        template_content = prompts.get(template_key, '')
        
        if not template_content:
            print(f"Warning: Prompt template content for key '{template_key}' not found. Using basic fallback.")
            if self.use_rag:
                template_content = "System: {system_message}\nContext: {context}\nQuestion: {question}\nAnswer:"
            else:
                template_content = "System: {system_message}\nQuestion: {question}\nAnswer:"
        
        return PromptTemplate.from_template(template_content)

    def _get_system_message(self, info_run: Dict) -> str:
        prompts_file = 'prompts.json' if self.use_rag else 'prompts_no_rag.json'
        path_prompts = os.path.join(os.path.dirname(__file__), prompts_file)
        
        with open(path_prompts, 'r', encoding='utf-8') as prompt_file:
            prompts = json.load(prompt_file)

        sys_mess = prompts.get('system_message', '')
        eval_attrs = prompts.get('evaluation-attributes', '')
        
        if eval_attrs:
            eval_attrs = eval_attrs.replace('REPLACE', info_run.get('Event Attributes', ''))
            eval_attrs = eval_attrs.replace('ACTIVITIES', str(info_run.get('Activities', '')))
            sys_mess += eval_attrs
            
        return sys_mess

    def produce_answer(self, question: str, vectdb, num_chunks: int, info_run: Dict) -> Tuple[str, str]:
        sys_mess = self._get_system_message(info_run)
        
        if self.use_rag:
            context = retrieve_context(vectdb, question, num_chunks)
            payload = {
                "question": question,
                "system_message": sys_mess,
                "context": context
            }
            result = self.chain.invoke(payload)
            
            if self.model_type == 'local':
                prompt, answer = self._parse_llm_answer(result)
            else:
                prompt = f'{sys_mess}\nHere is the most similar past traces: {context}\n' + f'Here is the prefix to predict: {question}\nAnswer: '
                answer = result.content if hasattr(result, 'content') else str(result)
        else:
            payload = {
                "question": question,
                "system_message": sys_mess
            }
            result = self.chain.invoke(payload)            
            if self.model_type == 'local':
                prompt, answer = self._parse_llm_answer(result)
            else:
                prompt = f'{sys_mess}\n' + f'Here is the prefix to predict: {question}\nAnswer: '
                answer = result.content if hasattr(result, 'content') else str(result)

        return prompt, answer

    def _parse_llm_answer(self, complete_answer: str) -> Tuple[str, str]:
        model_family_key = self.model_family.lower()
        delimiter = self.RESPONSE_DELIMITERS.get(model_family_key, 'Answer:')

        index = complete_answer.find(delimiter)
        if index != -1:
            prompt = complete_answer[:index + len(delimiter)]
            answer = complete_answer[index + len(delimiter):]
        else:
            prompt = complete_answer
            answer = ""

        return prompt, answer


EMBEDDING_MODELS = {
    'jina': [
        'jinaai/jina-embeddings-v4',
        'jinaai/jina-embeddings-v3',
        'jinaai/jina-embeddings-v2-base-en'
    ],
    'sentence_transformers': [
        'sentence-transformers/all-MiniLM-L12-v2',
        'sentence-transformers/all-MiniLM-L6-v2',
        'sentence-transformers/all-mpnet-base-v2'
    ]
}


def create_jina_v4_embeddings(model_name: str, device: str = 'cpu', default_task: str = 'retrieval'):
    class JinaV4Wrapper(HuggingFaceEmbeddings):
        def __init__(self):
            super().__init__(  # Dummy model to satisfy parent
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': device}
            )
            object.__setattr__(self, '_jina_model', SentenceTransformer(model_name, trust_remote_code=True, device=device))
            object.__setattr__(self, '_default_task', default_task)
            
        def embed_query(self, text: str) -> List[float]:   # Embed single query
            embedding = self._jina_model.encode(
                sentences=[text],
                task=self._default_task,
                prompt_name="query"
            )
            return embedding[0].tolist()
        
        def embed_documents(self, texts: List[str]) -> List[List[float]]: # Embed multiple document texts
            embeddings = self._jina_model.encode(
                sentences=texts,
                task=self._default_task,
                prompt_name="passage"
            )
            return embeddings.tolist()
    
    return JinaV4Wrapper()


def initialize_embedding_model(model_id: str, device: str = 'cpu', batch_size: int = 1) -> Tuple[object, int]:
    try:
        print(f"Initializing embedding model: {model_id}")
        if model_id == "jinaai/jina-embeddings-v4":
            print("Using custom JinaV4Embeddings wrapper")
            embedding_model = create_jina_v4_embeddings(model_id, device=device)
            test_embedding = embedding_model.embed_query("test")   # Test the model to get dimensions
            actual_dimension = len(test_embedding)
            print(f"Jina v4 embedding dimensions: {actual_dimension}")
            return embedding_model, actual_dimension
        
        fallback_models = []
        for model_family in EMBEDDING_MODELS.values():
            fallback_models.extend(model_family)
        models_to_try = [model_id] + [m for m in fallback_models if m != model_id]
        
        for model_name in models_to_try:
            try:
                print(f"Initializing embedding model: {model_name}")
                if model_name == "jinaai/jina-embeddings-v4":
                    print("Using custom JinaV4Embeddings wrapper for fallback")
                    embedding_model = create_jina_v4_embeddings(model_name, device=device)
                    test_embedding = embedding_model.embed_query("test")
                    actual_dimension = len(test_embedding)
                    print(f"Jina v4 embedding dimensions: {actual_dimension}")
                    return embedding_model, actual_dimension
                
                embedding_model = HuggingFaceEmbeddings(   # Standard model loading
                    model_name=model_name,
                    model_kwargs={'device': device})
                test_embedding = embedding_model.embed_query("test")   # Get dimensions
                actual_dimension = len(test_embedding)
                print(f"Successfully loaded {model_name} with {actual_dimension} dimensions")
                return embedding_model, actual_dimension
                
            except Exception as e:
                print(f"Failed to load {model_name}: {str(e)}")
                continue
        raise ValueError(f"Unable to load any embedding model. Last tried: {models_to_try}")
        
    except Exception as e:
        print(f"Error initializing embedding model: {str(e)}")
        raise

