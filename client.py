import asyncio
import json
import websockets
from typing import Optional, Dict, List, AsyncGenerator, Union
from enum import Enum
from loguru import logger

class ModelType(Enum):
    """모델 타입 정의"""
    DEEPSEEK = "deepseek"
    GEMMA = "gemma"
    LLAMA = "llama"
    MISTRAL = "mistral"
    CUSTOM = "custom"

class ConnectionStatus(Enum):
    """연결 상태 정의"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"

class BaseModelClient:
    """기본 모델 클라이언트"""
    def __init__(self, **kwargs):
        self.model_name = kwargs.get("model_name")
        self.websocket_url = kwargs.get("websocket_url", "ws://logosai.info:8765")
        self.model_type = kwargs.get("model_type")
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 4096)
        self.websocket = None
        self.status = ConnectionStatus.DISCONNECTED
        self._connected = False
        
    async def connect(self) -> bool:
        """서버 연결"""
        try:
            self.websocket = await websockets.connect(
                self.websocket_url,
                ping_interval=20,
                ping_timeout=10,
                max_size=10 * 1024 * 1024  # 10MB
            )
            self.status = ConnectionStatus.CONNECTED
            self._connected = True
            return True
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            self.status = ConnectionStatus.ERROR
            return False
    
    async def generate(self, prompt: str, stream: bool = False) -> Union[str, AsyncGenerator[Dict, None]]:
        """텍스트 생성"""
        if not self.websocket:
            if not await self.connect():
                raise ConnectionError("Failed to connect to server")
        
        request_data = {
            "type": "inference",
            "model_type": self.model_type.value,
            "model_name": self.model_name,
            "prompt": prompt,
            "stream": stream,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        await self.websocket.send(json.dumps(request_data))
        
        if stream:
            return self._stream_response()
        else:
            response = await self.websocket.recv()
            data = json.loads(response)
            
            if "error" in data:
                raise Exception(data["error"])
                
            return data.get("content", "")
    
    async def _stream_response(self) -> AsyncGenerator[Dict, None]:
        """스트리밍 응답 처리"""
        try:
            while True:
                response = await self.websocket.recv()
                data = json.loads(response)
                
                if "error" in data:
                    yield {"type": "error", "content": data["error"]}
                    break
                    
                content = data.get("content", "")
                if content:
                    yield {"type": "stream", "content": content}
                
                if data.get("done", False):
                    break
                    
        except Exception as e:
            yield {"type": "error", "content": str(e)}
    
    async def get_available_models(self) -> List[str]:
        """사용 가능한 모델 목록 조회"""
        if not self.websocket:
            if not await self.connect():
                raise ConnectionError("Failed to connect to server")
                
        await self.websocket.send(json.dumps({"type": "model_list"}))
        response = await self.websocket.recv()
        data = json.loads(response)
        return data.get("models", [])
    
    async def get_server_status(self) -> Dict:
        """서버 상태 조회"""
        if not self.websocket:
            if not await self.connect():
                raise ConnectionError("Failed to connect to server")
                
        await self.websocket.send(json.dumps({"type": "server_status"}))
        response = await self.websocket.recv()
        return json.loads(response)
    
    async def close(self):
        """연결 종료"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            self.status = ConnectionStatus.DISCONNECTED

class RemoteDeepSeek(BaseModelClient):
    """DeepSeek 모델 클라이언트"""
    def __init__(self, **kwargs):
        kwargs["model_type"] = ModelType.DEEPSEEK
        if "model_name" not in kwargs:
            kwargs["model_name"] = "deepseek-r1:8b"
        super().__init__(**kwargs)

class GemmaClient(BaseModelClient):
    """Gemma 모델용 클라이언트"""
    def __init__(self, **kwargs):
        kwargs["model_type"] = ModelType.GEMMA
        if "model_name" not in kwargs:
            kwargs["model_name"] = "gemma3:4b"  # 기본값 수정
        super().__init__(**kwargs)
        self.connection_lock = asyncio.Lock()
        self._last_request_id = 0
        self._response_futures = {}
        self._stream_queues = {}
        self._recv_lock = asyncio.Lock()
        self._connected = False
        
        logger.info(f"Initializing GemmaClient with model: {kwargs.get('model_name')}")
        
    async def connect(self) -> bool:
        async with self.connection_lock:
            if self._connected:
                return True
                
            try:
                self.websocket = await websockets.connect(
                    self.websocket_url,
                    ping_interval=20,
                    ping_timeout=10,
                    max_size=10 * 1024 * 1024
                )
                self._connected = True
                return True
            except Exception as e:
                logger.error(f"Gemma connection error: {str(e)}")
                return False
    
    async def generate(self, prompt: str, stream: bool = False, **kwargs) -> Union[str, AsyncGenerator[Dict, None]]:
        if not self._connected:
            await self.connect()
            
        request_id = self._last_request_id + 1
        self._last_request_id = request_id
        
        request_data = {
            "type": "inference",
            "model_type": "gemma",
            "model_name": self.model_name,
            "request_id": request_id,
            "prompt": prompt,
            "stream": stream,
            **kwargs
        }
        
        logger.info(f"Sending request with model: {self.model_name}")
        await self.websocket.send(json.dumps(request_data))
        
        if stream:
            async def stream_generator():
                try:
                    while True:
                        async with self._recv_lock:  # recv 호출 동기화
                            response = await self.websocket.recv()
                            data = json.loads(response)
                            
                            if data.get("type") == "error":
                                yield {"type": "error", "content": data.get("error")}
                                break
                                
                            yield {"type": "stream", "content": data.get("content", "")}
                            
                            if data.get("done", False):
                                break
                except Exception as e:
                    logger.error(f"Stream error: {str(e)}")
                    yield {"type": "error", "content": str(e)}
                    
            return stream_generator()
        else:
            async with self._recv_lock:  # recv 호출 동기화
                response = await self.websocket.recv()
                data = json.loads(response)
                return data.get("content", "")
    
    async def close(self):
        if self.websocket:
            await self.websocket.close()
            self._connected = False

def create_model_client(model_type: str, **kwargs) -> BaseModelClient:
    """모델 클라이언트 생성 팩토리 함수"""
    clients = {
        "deepseek": RemoteDeepSeek,
        "gemma": GemmaClient
    }
    
    client_class = clients.get(model_type.lower())
    if not client_class:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
    
    # model_type 파라미터 제거 (각 클래스에서 이미 설정됨)
    if "model_type" in kwargs:
        del kwargs["model_type"]
    
    return client_class(**kwargs)