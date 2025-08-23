"""HSemotion 패키지 초기화 모듈.

패키지 공개 API와 버전 정보를 정의합니다.
- `__all__`: 외부로 노출할 서브모듈 목록
- `__version__`: 패키지 버전 문자열
"""

__all__ = [
    "cli",
    "analyzer",
    "config",
    "emoji",
    "utils",
    "visualize",
]
__version__ = "0.1.0"