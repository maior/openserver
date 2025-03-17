import asyncio
import argparse
from typing import Dict, AsyncGenerator
from loguru import logger
import websockets
import json
import sys

async def chat_example(websocket_url: str):
    """간단한 채팅 예제"""
    prompts = [
            #"안녕하세요! 당신은 누구인가요?",
            "Question : 그리스는? 질문에 사용된 언어는 무엇인가? 응답을 줄때는 JSON으로 줘. {{ 'lang': 'en' or 'fr' or 'pt' }}",
        #"9.9와 9.11 중에서 어떤 수가 큰가?", # 답 9.9
        #"만약 5명의 친구가 각각 3개의 사과를 가지고 있고, 그중 2명이 각각 1개의 사과를 추가로 받았다면, 총 사과의 개수는 몇 개인가요? 또한, 각 친구가 가진 사과의 평균 개수는 얼마인가요?", # 답,  3.4개
        #"인공지능의 발전 방향에 대해 설명해주세요.",
        "한국의 전통 음식 중 가장 유명한 것은 무엇인가요?",
    ]
    
    try:
        # 웹소켓 연결
        async with websockets.connect(websocket_url) as websocket:
            for prompt in prompts:
                print(f"\n사용자: {prompt}")
                print("AI: ", end="", flush=True)
                
                # 요청 데이터 준비
                request_data = {
                    "type": "inference",
                    "prompt": prompt,
                    "stream": True,  # 스트리밍 모드 활성화
                    "model_name": "gemma-3-4b-it",  # 모델 이름 추가
                    "model_type": "gemma"  # 모델 타입 추가
                }
                
                # 요청 전송
                await websocket.send(json.dumps(request_data))
                
                # 응답 처리
                accumulated_text = ""  # 누적된 텍스트 저장
                while True:
                    try:
                        response = await websocket.recv()
                        data = json.loads(response)
                        
                        if "error" in data:
                            logger.error(f"Error: {data['error']}")
                            break
                        
                        if data.get("type") == "stream":
                            content = data.get("content", "")
                            if content:  # 내용이 있을 때만 출력
                                print(content, end="", flush=True)
                                accumulated_text += content
                            
                            # done이 True이면 스트리밍 완료
                            if data.get("done", False):
                                print()  # 줄바꿈
                                break
                        
                        elif data.get("type") == "response":
                            # 일반 응답 처리
                            print(data.get("content", ""))
                            break
                            
                    except websockets.exceptions.ConnectionClosed:
                        logger.error("WebSocket connection closed unexpectedly")
                        break
                    except Exception as e:
                        logger.error(f"Error processing response: {str(e)}")
                        break
                
                print("\n" + "-"*50)
                await asyncio.sleep(1)
                
    except websockets.exceptions.ConnectionClosed as e:
        logger.error(f"WebSocket connection closed: {e}")
    except Exception as e:
        logger.error(f"Error: {str(e)}")

async def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Hugging Face LLM Client Example")
    parser.add_argument("--url", default="ws://logosai.info:8765", help="WebSocket server URL")
    
    args = parser.parse_args()
    
    try:
        await chat_example(args.url)
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 
