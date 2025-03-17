import asyncio
import json
import uuid
import time
import websockets
from websockets.exceptions import ConnectionClosed
from typing import Dict, Optional, AsyncGenerator
import argparse
import sys
from loguru import logger
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from concurrent.futures import ThreadPoolExecutor
import os
from huggingface_hub import login

class ModelManager:
    """모델 관리자"""
    def __init__(self):
        self.models = {}
        self.client_models = {}  # 클라이언트별 모델 인스턴스 관리
        self.client_sessions = {}
        # self.executor = ThreadPoolExecutor(max_workers=3)
        self.executor = ThreadPoolExecutor(max_workers=torch.cuda.device_count() * 1)  # GPU 수에 맞춰 조정
        # self.executor = ThreadPoolExecutor(max_workers=torch.cuda.device_count() * 2)  # GPU 수에 맞춰 조정
        self.device_ids = list(range(torch.cuda.device_count()))
        self.current_device = 0
        self.last_cleanup_time = time.time()
        self.cleanup_interval = 300
        self.session_locks = {}  # 클라이언트별 락 추가
        self.model_locks = {}  # 모델별 락 관리
        self.client_states = {}  # 클라이언트 상태를 별도로 관리
        
        # Hugging Face 토큰 설정
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        logger.info(f"HUGGINGFACE_TOKEN: {self.hf_token}")
        if self.hf_token:
            login(self.hf_token)
        else:
            logger.warning("HUGGINGFACE_TOKEN environment variable is not set")

        # CUDA 메모리 관리 설정 추가
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        self.max_models_per_gpu = 1  # GPU당 최대 모델 수 제한
        self.model_cleanup_threshold = 0.8  # GPU 메모리 사용률 임계값 (80%)
        self.required_memory = 4 * 1024 * 1024 * 1024  # 4GB (모델당 필요한 최소 메모리)
        self.max_wait_time = 300  # 최대 대기 시간 (5분)
        self.memory_check_interval = 1  # 메모리 체크 간격 (1초)
        self.global_lock = asyncio.Lock()  # 글로벌 락 추가
        self.processing_clients = set()  # 현재 처리 중인 클라이언트 목록
        self.required_free_memory = 6 * 1024 * 1024 * 1024  # 6GB (필요한 여유 메모리)
        self.wait_queue = asyncio.Queue()  # 대기 큐 추가
        self.active_clients = {}  # 활성 클라이언트 상태 관리
        self.client_model_refs = {}  # 클라이언트별 모델 참조 카운트
        self.shared_models = {}  # 공유 모델 저장소 추가
        # 서버 시작 시 기본 모델 로드
        asyncio.create_task(self.initialize_default_model())

    async def initialize_default_model(self):
        """서버 시작 시 기본 모델 초기화"""
        try:
            model_name = "gemma-3-4b-it"
            model_type = "gemma"
            device_id = self.get_next_device()
            model_key = f"{model_type}:{model_name}"

            logger.info(f"Initializing default model: {model_name}")
            model_instance = await self.create_model_instance(model_name, device_id)
            self.shared_models[model_key] = model_instance
            logger.info(f"Default model initialized successfully on {model_instance['device']}")
        except Exception as e:
            logger.error(f"Error initializing default model: {str(e)}")

    async def get_client_lock(self, client_id: str):
        """클라이언트별 락 가져오기"""
        if client_id not in self.session_locks:
            self.session_locks[client_id] = asyncio.Lock()
        return self.session_locks[client_id]

    async def get_model_lock(self, model_key: str):
        if model_key not in self.model_locks:
            self.model_locks[model_key] = asyncio.Lock()
        return self.model_locks[model_key]

    def get_next_device(self):
        """다음 사용할 GPU 디바이스 선택"""
        device = self.device_ids[self.current_device]
        self.current_device = (self.current_device + 1) % len(self.device_ids)
        return device
        
    async def create_model_instance(self, model_name: str, device_id: int):
        """모델 인스턴스 생성"""
        try:
            device = f"cuda:{device_id}"
            logger.info(f"Creating model instance on {device}")
            
            # 모델 이름에 'google/' 접두어가 없으면 추가
            model_path = f"google/{model_name}" if not model_name.startswith("google/") else model_name
            logger.info(f"Loading model from: {model_path}")
            
            processor = AutoProcessor.from_pretrained(
                model_path,  # 수정: 하드코딩된 값 대신 model_path 사용
                token=self.hf_token,
                trust_remote_code=True,
                use_fast=True 
            )
            
            model = Gemma3ForConditionalGeneration.from_pretrained(
                model_path,  # 수정: 하드코딩된 값 대신 model_path 사용
                device_map=device,
                torch_dtype=torch.bfloat16,
                token=self.hf_token,
                trust_remote_code=True
            ).eval()
            
            return {
                "model": model,
                "processor": processor,
                "device": device,
                "type": "gemma",
                "last_used": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error creating model instance: {str(e)}")
            raise

    async def wait_for_memory(self, device_id: int) -> bool:
        """충분한 메모리가 확보될 때까지 대기"""
        start_time = time.time()
        while True:
            try:
                free_memory = torch.cuda.get_device_properties(device_id).total_memory - torch.cuda.memory_allocated(device_id)
                memory_usage = torch.cuda.memory_allocated(device_id) / torch.cuda.get_device_properties(device_id).total_memory
                
                logger.info(f"GPU {device_id} - Free memory: {free_memory / 1024**3:.2f}GB, Usage: {memory_usage:.2%}")
                
                if free_memory >= self.required_memory:
                    return True
                
                # 최대 대기 시간 초과 확인
                if time.time() - start_time > self.max_wait_time:
                    logger.warning(f"Maximum wait time ({self.max_wait_time}s) exceeded for GPU {device_id}")
                    return False
                
                # 메모리 정리 시도
                await self.cleanup_unused_models()
                
                # 1초 대기 후 재시도
                await asyncio.sleep(self.memory_check_interval)
                
            except Exception as e:
                logger.error(f"Error checking GPU memory: {str(e)}")
                return False

    async def check_and_wait_for_resources(self, client_id: str, device_id: int) -> bool:
        """리소스 사용 가능 여부 확인 및 대기"""
        while True:
            async with self.global_lock:
                # 현재 메모리 상태 확인
                free_memory = torch.cuda.get_device_properties(device_id).total_memory - torch.cuda.memory_allocated(device_id)
                
                # 처리 중인 클라이언트가 없고 충분한 메모리가 있는 경우
                if (len(self.processing_clients) == 0 and 
                    free_memory >= self.required_free_memory):
                    self.processing_clients.add(client_id)
                    logger.info(f"[Client {client_id}] Resources allocated. Free memory: {free_memory/1024**3:.2f}GB")
                    return True
                
                # 이미 처리 중인 클라이언트가 있는 경우
                if len(self.processing_clients) > 0:
                    logger.info(f"[Client {client_id}] Waiting for other clients to finish. Active: {self.processing_clients}")
                    await asyncio.sleep(1)
                    continue
                
                # 메모리가 부족한 경우
                if free_memory < self.required_free_memory:
                    logger.info(f"[Client {client_id}] Insufficient memory. Available: {free_memory/1024**3:.2f}GB, Required: {self.required_free_memory/1024**3:.2f}GB")
                    await asyncio.sleep(1)
                    continue

    async def release_resources(self, client_id: str):
        """리소스 해제"""
        async with self.global_lock:
            if client_id in self.processing_clients:
                self.processing_clients.remove(client_id)
                logger.info(f"[Client {client_id}] Resources released")
                await self.cleanup_unused_models()

    async def register_client(self, client_id: str):
        """클라이언트 등록"""
        self.active_clients[client_id] = {
            "connected": True,
            "model_key": None,
            "last_activity": time.time()
        }
        logger.info(f"[Client {client_id}] Registered new client")

    async def unregister_client(self, client_id: str):
        """클라이언트 등록 해제"""
        if client_id in self.active_clients:
            model_key = self.active_clients[client_id]["model_key"]
            if model_key:
                # 모델 참조 카운트 감소
                self.client_model_refs[model_key] = self.client_model_refs.get(model_key, 0) - 1
                if self.client_model_refs[model_key] <= 0:
                    # 해당 모델을 사용하는 클라이언트가 없으면 정리
                    await self.cleanup_model(model_key)
            
            del self.active_clients[client_id]
            logger.info(f"[Client {client_id}] Unregistered client")

    async def cleanup_model(self, model_key: str):
        """특정 모델 정리"""
        if model_key in self.client_models:
            model_info = self.client_models[model_key]
            if "model" in model_info:
                model_info["model"].cpu()
                del model_info["model"]
            if "processor" in model_info:
                del model_info["processor"]
            del self.client_models[model_key]
            if model_key in self.client_model_refs:
                del self.client_model_refs[model_key]
            torch.cuda.empty_cache()
            logger.info(f"Cleaned up model: {model_key}")

    async def get_or_create_model(self, model_name: str, model_type: str = None, client_id: str = None):
        """모델 가져오기 (공유 모델 우선 사용)"""
        model_key = f"{model_type}:{model_name}"
        
        # 공유 모델이 있으면 사용
        if model_key in self.shared_models:
            model_info = self.shared_models[model_key]
            model_info["last_used"] = time.time()
            if client_id in self.active_clients:
                self.active_clients[client_id]["model_key"] = model_key
            return model_info

        # 공유 모델이 없는 경우에만 새로 생성
        if model_key not in self.shared_models:
            device_id = self.get_next_device()
            await self.check_and_wait_for_resources(client_id, device_id)
            
            model_instance = await self.create_model_instance(model_name, device_id)
            self.shared_models[model_key] = model_instance
            
            if client_id in self.active_clients:
                self.active_clients[client_id]["model_key"] = model_key
            
            logger.info(f"Created new shared model instance: {model_key}")
        
        return self.shared_models[model_key]

    async def generate(self, model_name: str, prompt: str, stream: bool = False, model_type: str = None, client_id: str = None) -> AsyncGenerator[Dict, None]:
        """텍스트 생성"""
        if client_id not in self.active_clients:
            raise RuntimeError("Client not registered")

        try:
            model_key = f"{client_id}:{model_type}:{model_name}"
            async with await self.get_model_lock(model_key):
                model_info = await self.get_or_create_model(model_name, model_type, client_id)
                model = model_info["model"]
                processor = model_info["processor"]
                
                # 클라이언트별 상태 초기화
                if client_id not in self.client_states:
                    self.client_states[client_id] = {
                        "current_input_ids": None,
                        "current_attention_mask": None
                    }
                client_state = self.client_states[client_id]
                
                logger.info(f"[Client {client_id}] Model loaded successfully on {model_info['device']}")
                
                # 메시지 형식 준비
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant."}]
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}]
                    }
                ]
                
                logger.info(f"[Client {client_id}] Processing input...")
                # 입력 처리
                inputs = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt"
                    )
                )
                
                # 입력을 GPU로 이동 및 dtype 설정
                # input_ids는 Long 타입으로 유지하고, 다른 텐서들만 bfloat16으로 변환
                inputs = {
                    "input_ids": inputs["input_ids"].to(model.device),
                    "attention_mask": inputs["attention_mask"].to(model.device) if "attention_mask" in inputs else None
                }
                
                # 모델의 임베딩 레이어 dtype 확인 및 로깅
                logger.info(f"[Client {client_id}] Model embedding dtype: {model.get_input_embeddings().weight.dtype}")
                logger.info(f"[Client {client_id}] Input IDs dtype: {inputs['input_ids'].dtype}")
                if inputs.get("attention_mask") is not None:
                    logger.info(f"[Client {client_id}] Attention mask dtype: {inputs['attention_mask'].dtype}")
                
                input_len = inputs["input_ids"].shape[-1]
                logger.info(f"[Client {client_id}] Input length: {input_len} tokens")
                
                if stream:
                    logger.info(f"[Client {client_id}] Starting streaming generation...")
                    # 클라이언트 상태 업데이트
                    client_state["current_input_ids"] = inputs["input_ids"]
                    client_state["current_attention_mask"] = inputs.get("attention_mask")
                    
                    total_new_tokens = 0
                    batch_size = 20
                    
                    try:
                        while total_new_tokens < 1000000:
                            with torch.inference_mode():
                                generation_inputs = {
                                    "input_ids": client_state["current_input_ids"].contiguous(),  # 입력 텐서에 .contiguous() 메서드를 추가하여 메모리가 연속적으로 할당되도록 보장
                                    "attention_mask": client_state["current_attention_mask"],
                                    "max_new_tokens": batch_size,
                                    "do_sample": True,
                                    "temperature": 0.7,
                                    # "top_p": 0.95,  # 추가: 상위 확률 샘플링
                                    # "top_k": 50,    # 추가: 상위 k개 토큰으로 제한
                                    "pad_token_id": processor.tokenizer.pad_token_id,
                                    "num_return_sequences": 1,
                                    "repetition_penalty": 1.1,
                                    # "no_repeat_ngram_size": 3,  # 추가: n-gram 반복 방지
                                    "early_stopping": False
                                }
                                
                                outputs = await asyncio.get_event_loop().run_in_executor(
                                    self.executor,
                                    lambda: model.generate(**{k: v for k, v in generation_inputs.items() if v is not None})
                                )
                            
                            new_tokens = outputs[0, client_state["current_input_ids"].size(1):]
                            text = processor.decode(new_tokens, skip_special_tokens=True)
                            current_batch_tokens = len(new_tokens)
                            total_new_tokens += current_batch_tokens
                            
                            if text:
                                yield {
                                    "type": "stream",
                                    "content": text,
                                    "model_type": model_info["type"]
                                }
                            
                            if processor.tokenizer.eos_token_id in new_tokens or current_batch_tokens < batch_size:
                                break
                            
                            # 클라이언트 상태 업데이트
                            client_state["current_input_ids"] = outputs
                            if client_state["current_attention_mask"] is not None:
                                client_state["current_attention_mask"] = torch.ones(
                                    (outputs.size(0), outputs.size(1)),
                                    dtype=client_state["current_attention_mask"].dtype,
                                    device=client_state["current_attention_mask"].device
                                )
                            
                            del outputs
                            del new_tokens
                            torch.cuda.empty_cache()
                            
                        # 클라이언트 상태 정리
                        del client_state["current_input_ids"]
                        del client_state["current_attention_mask"]
                        client_state["current_input_ids"] = None
                        client_state["current_attention_mask"] = None
                        
                        yield {
                            "type": "stream",
                            "content": "",
                            "done": True,
                            "model_type": model_info["type"]
                        }
                    finally:
                        torch.cuda.empty_cache()
                else:
                    logger.info(f"[Client {client_id}] Starting single generation...")
                    # 일반 생성
                    with torch.inference_mode():
                        outputs = await asyncio.get_event_loop().run_in_executor(
                            self.executor,
                            lambda: model.generate(
                                **inputs,
                                max_new_tokens=1000,  # 토큰 제한을 1000으로 증가
                                do_sample=True,
                                temperature=0.7,
                                repetition_penalty=1.1,  # 반복 방지를 위한 페널티 추가
                                pad_token_id=processor.tokenizer.pad_token_id,
                                num_return_sequences=1
                            )
                        )
                    
                    # 새로 생성된 토큰 디코딩
                    new_tokens = outputs[0, input_len:]
                    text = processor.decode(new_tokens, skip_special_tokens=True)
                    
                    logger.info(f"[Client {client_id}] Generation completed. Output length: {len(new_tokens)} tokens")
                    logger.info(f"[Client {client_id}] Generated text: {text[:100]}...")
                    
                    # GPU 메모리 정리
                    del outputs
                    del new_tokens
                    torch.cuda.empty_cache()
                    
                    yield {
                        "type": "response",
                        "content": text,
                        "model_type": model_info["type"]
                    }
                
                # 생성 완료 후 마지막 사용 시간 업데이트
                model_info["last_used"] = time.time()
                self.active_clients[client_id]["last_activity"] = time.time()

        except Exception as e:
            logger.error(f"[Client {client_id}] Generation error: {str(e)}")
            yield {"type": "error", "content": str(e)}

    async def cleanup_memory(self):
        """주기적인 메모리 정리"""
        current_time = time.time()
        if current_time - self.last_cleanup_time > self.cleanup_interval:
            logger.info("Performing periodic memory cleanup...")
            torch.cuda.empty_cache()
            self.last_cleanup_time = current_time

    async def cleanup_client_session(self, client_id: str):
        """클라이언트 세션 정리"""
        try:
            # 리소스 해제
            await self.release_resources(client_id)
            
            # 클라이언트의 모델 인스턴스 정리
            models_to_remove = [
                key for key in self.client_models.keys()
                if key.startswith(f"{client_id}:")
            ]
            
            for key in models_to_remove:
                if key in self.client_models:
                    logger.info(f"Removing model for disconnected client: {key}")
                    model_info = self.client_models[key]
                    
                    # GPU 메모리에서 모델 제거
                    if "model" in model_info:
                        model_info["model"].cpu()  # GPU에서 CPU로 이동
                        del model_info["model"]
                    if "processor" in model_info:
                        del model_info["processor"]
                    
                    del self.client_models[key]
            
            # 기존 세션 정리 로직
            if client_id in self.client_sessions:
                del self.client_sessions[client_id]
            if client_id in self.session_locks:
                del self.session_locks[client_id]
            if client_id in self.client_states:
                del self.client_states[client_id]
            
            # GPU 메모리 정리
            torch.cuda.empty_cache()
            logger.info(f"Cleaned up session and freed GPU memory for client {client_id}")
            
        except Exception as e:
            logger.error(f"Error during client session cleanup: {str(e)}")

    async def cleanup_unused_models(self):
        """사용하지 않는 모델 정리 (공유 모델은 제외)"""
        try:
            current_time = time.time()
            models_to_remove = []
            
            # 클라이언트별 모델만 정리 (공유 모델은 유지)
            for client_model_key, model_info in self.client_models.items():
                if current_time - model_info["last_used"] > 300:  # 5분
                    models_to_remove.append(client_model_key)
            
            for key in models_to_remove:
                if key not in self.shared_models:  # 공유 모델은 제외
                    logger.info(f"Removing unused client model: {key}")
                    await self.cleanup_model(key)
            
            # 메모리 상태 로깅
            if models_to_remove:
                torch.cuda.empty_cache()
                logger.info("Cleaned up unused client models")
                
        except Exception as e:
            logger.error(f"Error during model cleanup: {str(e)}")

class WebSocketServer:
    """WebSocket 서버"""
    def __init__(self, host: str = "0.0.0.0", port: int = 8765, stream: bool = False):
        self.host = host
        self.port = port
        self.stream = stream
        self.model_manager = ModelManager()
        self.active_connections = set()
        self.client_queues = {}  # 클라이언트별 응답 큐 추가
    
    async def handle_client(self, websocket):
        client_id = str(uuid.uuid4())
        try:
            # 클라이언트 등록
            await self.model_manager.register_client(client_id)
            self.active_connections.add(websocket)
            self.client_queues[client_id] = asyncio.Queue()

            logger.info(f"[Client {client_id}] New client connected")
            logger.info(f"[Client {client_id}] Active connections: {len(self.active_connections)}")
            logger.info(f"[Client {client_id}] Streaming mode: {self.stream}")
            
            # 응답 전송 태스크 생성
            response_task = asyncio.create_task(self.handle_responses(websocket, client_id))
            
            async for message in websocket:
                logger.info(f"[Client {client_id}] Received message: {message[:200]}...")
                await self.process_message(websocket, message, client_id)
        except ConnectionClosed:
            logger.info(f"[Client {client_id}] Client disconnected")
        except Exception as e:
            logger.error(f"[Client {client_id}] Error handling client: {str(e)}")
            logger.error(f"[Client {client_id}] Error traceback:", exc_info=True)
        finally:
            # 정리 작업
            response_task.cancel()
            self.active_connections.remove(websocket)
            del self.client_queues[client_id]
            await self.model_manager.cleanup_client_session(client_id)
            logger.info(f"[Client {client_id}] Connection and session cleaned up")
            logger.info(f"[Client {client_id}] Remaining active connections: {len(self.active_connections)}")
    
    async def handle_responses(self, websocket, client_id: str):
        """클라이언트별 응답 처리"""
        try:
            while True:
                response = await self.client_queues[client_id].get()
                try:
                    await websocket.send(json.dumps(response))
                except Exception as e:
                    logger.error(f"[Client {client_id}] Error sending response: {str(e)}")
                    break
        except asyncio.CancelledError:
            logger.info(f"[Client {client_id}] Response handler cancelled")
        except Exception as e:
            logger.error(f"[Client {client_id}] Response handler error: {str(e)}")
    
    async def process_message(self, websocket, message: str, client_id: str):
        """메시지 처리"""
        try:
            data = json.loads(message)
            request_type = data.get("type")
            
            if request_type == "inference":
                # 서버의 스트리밍 설정을 우선적으로 사용
                data["stream"] = self.stream if self.stream else data.get("stream", False)
                await self.handle_inference(websocket, data, client_id)
            elif request_type == "model_list":
                await self.handle_model_list(websocket, client_id)
            elif request_type == "server_status":
                await self.handle_server_status(websocket, client_id)
            else:
                error_msg = f"Unknown request type: {request_type}"
                logger.error(f"[Client {client_id}] {error_msg}")
                await websocket.send(json.dumps({
                    "error": error_msg
                }))
                
        except json.JSONDecodeError as e:
            logger.error(f"[Client {client_id}] Invalid JSON message: {str(e)}")
            await websocket.send(json.dumps({
                "error": "Invalid JSON message"
            }))
        except Exception as e:
            logger.error(f"[Client {client_id}] Error processing message: {str(e)}")
            await websocket.send(json.dumps({
                "error": str(e)
            }))
    
    async def handle_inference(self, websocket, data: Dict, client_id: str):
        """추론 요청 처리"""
        try:
            prompt = data.get("prompt")
            if not prompt:
                raise ValueError("Prompt is required")
            
            model_name = data.get("model_name", "gemma-3-4b-it")
            model_type = data.get("model_type", "gemma")
            stream = data.get("stream", False)
            
            logger.info(f"[Client {client_id}] Processing inference request:")
            logger.info(f"[Client {client_id}] - Stream mode: {stream}")
            
            async for response in self.model_manager.generate(
                model_name=model_name,
                prompt=prompt,
                stream=stream,
                model_type=model_type,
                client_id=client_id
            ):
                # 응답을 클라이언트 큐에 추가
                await self.client_queues[client_id].put(response)
                
        except Exception as e:
            error_msg = f"Inference error: {str(e)}"
            logger.error(f"[Client {client_id}] {error_msg}")
            await self.client_queues[client_id].put({
                "error": error_msg
            })
    
    async def handle_model_list(self, websocket, client_id: str):
        """모델 목록 요청 처리"""
        models = ["gemma-3-4b-it"]  # 현재 지원하는 모델
        await websocket.send(json.dumps({
            "type": "model_list",
            "models": models
        }))
    
    async def handle_server_status(self, websocket, client_id: str):
        """서버 상태 요청 처리"""
        status = {
            "type": "server_status",
            "active_connections": len(self.active_connections),
            "loaded_models": list(self.model_manager.models.keys()),
            "available_gpus": len(self.model_manager.device_ids)
        }
        await websocket.send(json.dumps(status))
    
    async def start(self):
        """서버 시작"""
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        logger.info(f"Available GPUs: {torch.cuda.device_count()}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            max_size=10 * 1024 * 1024,  # 10MB
            ping_interval=20,
            ping_timeout=10
        )
        
        logger.info("WebSocket server is running and ready to accept connections")
        await server.wait_closed()

async def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Hugging Face LLM Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind")
    parser.add_argument("--stream", action="store_true", help="Enable streaming mode for all requests")
    
    args = parser.parse_args()
    
    # 서버 시작 로그
    logger.info("Starting Hugging Face LLM Server with following configuration:")
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Streaming mode: {args.stream}")
    
    server = WebSocketServer(args.host, args.port, args.stream)
    await server.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {str(e)}") 