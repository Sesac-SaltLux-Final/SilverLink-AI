import os
from typing import List

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

ENV: str = ""


class Configs(BaseSettings):
    # base
    ENV: str = os.getenv("ENV", "dev")
    API: str = "/api"
    API_STR: str = "/api"
    # API_V2_STR: str = "/api/v2"
    PROJECT_NAME: str = "SilverLink AI API"
    # ENV_DATABASE_MAPPER: dict = {
    #     "prod": "fca",
    #     "stage": "stage-fca",
    #     "dev": "dev-fca",
    #     "test": "test-fca",
    # }
    # DB_ENGINE_MAPPER: dict = {
    #     "postgresql": "postgresql",
    #     "mysql": "mysql+pymysql",
    # }

    PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    PORT: int = 5000

    # date
    DATETIME_FORMAT: str = "%Y-%m-%dT%H:%M:%S"
    DATE_FORMAT: str = "%Y-%m-%d"

    # auth
    # SECRET_KEY: str = os.getenv("SECRET_KEY", "")
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["*"]

    # --- Chatbot Configs ---
    # OpenAI
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # Milvus / Zilliz
    MILVUS_URI: str = os.getenv("MILVUS_URI", "")
    MILVUS_TOKEN: str = os.getenv("MILVUS_TOKEN", "")
    FAQ_COLLECTION_NAME: str = os.getenv("FAQ_COLLECTION_NAME", "faq_collection")
    INQUIRY_COLLECTION_NAME: str = os.getenv("INQUIRY_COLLECTION_NAME", "inquiry_collection")

    # Spring Boot backend
    SPRING_BOOT_URL: str = os.getenv("SPRING_BOOT_URL", "http://localhost:8080")
    SPRING_BOOT_API_TOKEN: str = os.getenv("SPRING_BOOT_API_TOKEN", "")

    # database
    # DB: str = os.getenv("DB", "postgresql")
    # DB_USER: str = os.getenv("DB_USER")
    # DB_PASSWORD: str = os.getenv("DB_PASSWORD")
    # DB_HOST: str = os.getenv("DB_HOST")
    # DB_PORT: str = os.getenv("DB_PORT", "3306")
    # DB_ENGINE: str = DB_ENGINE_MAPPER.get(DB, "postgresql")

    # DATABASE_URI_FORMAT: str = "{db_engine}://{user}:{password}@{host}:{port}/{database}"

    # DATABASE_URI = "{db_engine}://{user}:{password}@{host}:{port}/{database}".format(
    #     db_engine=DB_ENGINE,
    #     user=DB_USER,
    #     password=DB_PASSWORD,
    #     host=DB_HOST,
    #     port=DB_PORT,
    #     database=ENV_DATABASE_MAPPER[ENV],
    # )

    # find query
    # PAGE = 1
    # PAGE_SIZE = 20
    # ORDERING = "-id"
    
    # server_url
    CALL_CONTROLL_URL:str = os.getenv("CALL_CONTROLL_URL")
    
    # llm
    INFERENCE_MODEL:str = "gpt-4o-mini"
    
    # api_key
    OPENAI_API_KEY:str = os.getenv("OPENAI_API_KEY")
    LUXIA_API_KEY :str= os.getenv("LUXIA_API_KEY")
    # CLOVA_STT_URL:str = os.getenv("CLOVA_STT_URL")
    # CLOVA_SECRET_KEY:str = os.getenv("CLOVA_SECRET_KEY")
    
    TWILIO_SID:str = os.getenv("TWILIO_SID")
    TWILIO_TOKEN:str = os.getenv("TWILIO_TOKEN")
    
    # number
    SILVERLINK_NUMBER:str = os.getenv("SILVERLINK_NUMBER")
    NUMBER:str = ''
    
    # AWS SQS Configuration
    AWS_REGION: str = os.getenv("AWS_REGION", "ap-northeast-2")
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    SQS_QUEUE_URL: str = os.getenv("SQS_QUEUE_URL", "")
    SQS_DLQ_URL: str = os.getenv("SQS_DLQ_URL", "")
    AWS_S3_BUCKET_NAME: str = os.getenv("AWS_S3_BUCKET_NAME", "silverlink-storage")

    class Config:
        case_sensitive = True


class TestConfigs(Configs):
    ENV: str = "test"


configs = Configs()

if ENV == "prod":
    pass
elif ENV == "stage":
    pass
elif ENV == "test":
    setting = TestConfigs()
