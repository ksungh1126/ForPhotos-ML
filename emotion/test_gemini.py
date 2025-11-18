#!/usr/bin/env python3
"""Gemini API 연결 테스트 스크립트"""

import os
import sys
from pathlib import Path

# .env 파일 로드
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ .env 파일 로드: {env_path}\n")
    else:
        print(f"⚠️  .env 파일 없음: {env_path}")
        print("   환경 변수 또는 직접 설정을 사용합니다.\n")
except ImportError:
    print("⚠️  python-dotenv가 설치되지 않았습니다.")
    print("   pip install python-dotenv\n")

def test_gemini_api():
    """Gemini API 연결 및 기능 테스트"""

    print("=" * 60)
    print("Gemini API 테스트")
    print("=" * 60)

    # 1. API 키 확인
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("\n설정 방법:")
        print("  1. .env 파일 생성:")
        print("     cp emotion/.env.example emotion/.env")
        print("  2. .env 파일에 API 키 입력:")
        print("     GEMINI_API_KEY=your_api_key_here")
        print("\n  또는 직접 환경 변수 설정:")
        print("     export GEMINI_API_KEY='YOUR_API_KEY_HERE'")
        print("\nAPI 키 발급: https://makersuite.google.com/app/apikey")
        sys.exit(1)

    print(f"✅ API 키 확인: {api_key[:10]}...{api_key[-4:]}")

    # 2. 모듈 import 테스트
    try:
        from emotion.recommender.gemini_generator import GeminiRecommendationEngine, RecommendationRequest
        print("✅ Gemini 모듈 import 성공")
    except ImportError as e:
        print(f"❌ 모듈 import 실패: {e}")
        print("\n해결 방법:")
        print("  pip install requests")
        sys.exit(1)

    # 3. 엔진 초기화 테스트
    try:
        engine = GeminiRecommendationEngine(api_key=api_key)
        print("✅ Gemini 엔진 초기화 성공")
    except Exception as e:
        print(f"❌ 엔진 초기화 실패: {e}")
        sys.exit(1)

    # 4. 테스트 이미지 확인
    test_images = [
        Path("/home/work/wonjun/ForPhotos-ML/emotion/photo1.jpg"),
        Path("/home/work/wonjun/ForPhotos-ML/emotion/photo2.jpg"),
    ]

    test_image = None
    for img in test_images:
        if img.exists():
            test_image = img
            break

    if not test_image:
        print("⚠️  테스트 이미지를 찾을 수 없습니다.")
        print("  직접 이미지 경로를 지정하세요:")
        print("  python test_gemini.py <image_path>")

        if len(sys.argv) > 1:
            test_image = Path(sys.argv[1])
            if not test_image.exists():
                print(f"❌ 이미지 파일을 찾을 수 없습니다: {test_image}")
                sys.exit(1)
        else:
            sys.exit(1)

    print(f"✅ 테스트 이미지: {test_image}")

    # 5. API 호출 테스트
    print("\n📡 Gemini API 호출 중...")
    try:
        request = RecommendationRequest(
            image_path=test_image,
            user_hint="친구들과 즐거운 시간"
        )

        result = engine.generate(request)

        print("✅ API 호출 성공!")
        print("\n" + "=" * 60)
        print("결과:")
        print("=" * 60)
        print(f"\n📝 SNS 캡션:")
        print(f"  {result.caption}")
        print(f"\n#️⃣ 해시태그:")
        print(f"  {', '.join('#' + tag for tag in result.hashtags)}")
        print(f"\n🎵 추천 음악:")
        print(f"  {result.song_title} - {result.song_artist}")
        print("\n" + "=" * 60)

    except Exception as e:
        print(f"❌ API 호출 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n✅ 모든 테스트 통과!")
    print("\n다음 단계:")
    print("  1. .env 파일에 USE_GEMINI_API=true 추가:")
    print("     echo 'USE_GEMINI_API=true' >> emotion/.env")
    print("  2. 서버 실행:")
    print("     uvicorn emotion.api.server:app --host 0.0.0.0 --port 8000")
    print("\n  또는 환경 변수로 직접 설정:")
    print("     USE_GEMINI_API=true uvicorn emotion.api.server:app --port 8000")


if __name__ == "__main__":
    test_gemini_api()
