import os
from dotenv import load_dotenv

# .envファイルを読み込む
load_dotenv()

# OpenAI設定
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4')
OPENAI_ORGANIZATION = os.getenv('OPENAI_ORGANIZATION')

# 設定の検証
def validate_config():
    """設定が正しく読み込まれているかチェック"""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEYが設定されていません。.envファイルを確認してください。")
    return True 