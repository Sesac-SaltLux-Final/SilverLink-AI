from dependency_injector import containers, providers

# from app.core.database import Database
from app.callbot.repository import CallbotRepository
from app.chatbot.repository import ChatbotRepository
from app.ocr.repository import OcrRepository
from app.callbot.services import CallbotService
from app.chatbot.services.chatbot_service import ChatbotService
from app.chatbot.services.data_sync_service import DataSyncService
from app.ocr.services import OcrService
from app.core.config import configs
from app.integration.llm.openai_client import LLM
from app.integration.tts.luxia_client import TTS
from app.integration.call import CALL
from app.queue.sqs_client import SQSClient
from app.queue.worker import SQSWorker
from app.queue.dlq_handler import DLQHandler


class Container(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(
        modules=[
            "app.api.endpoints.callbot",
            "app.api.endpoints.chatbot",
            "app.api.endpoints.ocr",
        ]
    )

    # db = providers.Singleton(Database, db_url=configs.DATABASE_URI)
    llm = providers.Singleton(LLM, model_version=configs.INFERENCE_MODEL, api_key=configs.OPENAI_API_KEY)
    # stt = providers.Singleton(STT, model_name="naver", secret_key=configs.CLOVA_SECRET_KEY, url=configs.CLOVA_STT_URL)
    tts = providers.Singleton(TTS, api_key=configs.LUXIA_API_KEY, url=configs.CALL_CONTROLL_URL)
    call = providers.Singleton(CALL, account_sid=configs.TWILIO_SID, auth_token=configs.TWILIO_TOKEN, url=configs.CALL_CONTROLL_URL, number=configs.NUMBER, silverlink_number=configs.SILVERLINK_NUMBER)
    
    # AWS SQS
    sqs_client = providers.Singleton(
        SQSClient,
        queue_url=configs.SQS_QUEUE_URL,
        dlq_url=configs.SQS_DLQ_URL,
        region_name=configs.AWS_REGION,
        aws_access_key_id=configs.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=configs.AWS_SECRET_ACCESS_KEY
    )
     
    callbot_repository = providers.Factory(CallbotRepository)
    chatbot_repository = providers.Factory(ChatbotRepository)
    ocr_repository = providers.Factory(OcrRepository)

    callbot_service = providers.Factory(CallbotService, callbot_repository=callbot_repository, llm=llm, call=call, tts=tts)
    datasync_service = providers.Factory(DataSyncService, chatbot_repository=chatbot_repository)
    chatbot_service = providers.Factory(ChatbotService, chatbot_repository=chatbot_repository)
    ocr_service = providers.Factory(OcrService, ocr_repository=ocr_repository, llm=llm)
    
    # SQS Worker & DLQ Handler
    sqs_worker = providers.Factory(
        SQSWorker,
        sqs_client=sqs_client,
        callbot_service=callbot_service,
        call_client=call
    )
    dlq_handler = providers.Factory(DLQHandler, sqs_client=sqs_client)