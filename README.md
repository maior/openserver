# OpenServer

WebSocket 기반 LLM 서버

## 설치 방법

```bash
pip install .
```

## 사용 방법

### 서버 실행
```bash
openserver --host 0.0.0.0 --port 8765 --stream
```

### 클라이언트 실행
```bash
openserver-client --url ws://localhost:8765
```

## 환경 변수

- `HUGGINGFACE_TOKEN`: Hugging Face 토큰 (필수) 
- `PYTORCH_CUDA_ALLOC_CONF`: CUDA 메모리 할당 설정

## 요구사항

- Python >= 3.8 
- CUDA 지원 GPU (선택사항) 

## Example Code
from openserver.client import GemmaClient
from openserver.config import ServerConfig
