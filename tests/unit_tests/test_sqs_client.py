# """
# Unit Tests for SQS Queue Module

# SQS 클라이언트, 메시지 스키마, 워커의 단위 테스트
# """
# import pytest
# from unittest.mock import Mock, patch, MagicMock
# from datetime import datetime
# import json

# from app.queue.message_schema import (
#     CallRequestMessage,
#     CallResultMessage,
#     DLQMessage,
#     CallStatus
# )
# from app.queue.sqs_client import SQSClient
# from app.queue.worker import SQSWorker
# from app.queue.dlq_handler import DLQHandler


# class TestMessageSchema:
#     """메시지 스키마 테스트"""
    
#     def test_call_request_message_creation(self):
#         """통화 요청 메시지 생성 테스트"""
#         msg = CallRequestMessage(
#             message_id="test-123",
#             schedule_id=1,
#             elderly_id=100,
#             elderly_name="홍길동",
#             phone_number="+821012345678",
#             scheduled_time=datetime.utcnow()
#         )
        
#         assert msg.message_id == "test-123"
#         assert msg.schedule_id == 1
#         assert msg.elderly_name == "홍길동"
#         assert msg.retry_count == 0  # 기본값
    
#     def test_call_request_message_serialization(self):
#         """메시지 JSON 직렬화 테스트"""
#         msg = CallRequestMessage(
#             message_id="test-456",
#             schedule_id=2,
#             elderly_id=200,
#             elderly_name="김철수",
#             phone_number="+821098765432",
#             scheduled_time=datetime(2026, 1, 29, 10, 0, 0)
#         )
        
#         json_str = msg.model_dump_json()
#         parsed = json.loads(json_str)
        
#         assert parsed["message_id"] == "test-456"
#         assert parsed["schedule_id"] == 2
#         assert "2026-01-29" in parsed["scheduled_time"]
    
#     def test_call_result_message_success(self):
#         """통화 결과 메시지 (성공) 테스트"""
#         result = CallResultMessage(
#             message_id="test-123",
#             schedule_id=1,
#             call_sid="CA123456",
#             status=CallStatus.COMPLETED,
#             duration_seconds=120
#         )
        
#         assert result.status == CallStatus.COMPLETED
#         assert result.call_sid == "CA123456"
#         assert result.error_message is None
    
#     def test_call_result_message_failure(self):
#         """통화 결과 메시지 (실패) 테스트"""
#         result = CallResultMessage(
#             message_id="test-123",
#             schedule_id=1,
#             status=CallStatus.FAILED,
#             error_message="Connection timeout"
#         )
        
#         assert result.status == CallStatus.FAILED
#         assert result.error_message == "Connection timeout"
#         assert result.call_sid is None
    
#     def test_dlq_message_creation(self):
#         """DLQ 메시지 생성 테스트"""
#         original = CallRequestMessage(
#             message_id="test-789",
#             schedule_id=3,
#             elderly_id=300,
#             elderly_name="박영희",
#             phone_number="+821011112222",
#             scheduled_time=datetime.utcnow(),
#             retry_count=3
#         )
        
#         dlq_msg = DLQMessage(
#             original_message=original,
#             failure_reason="Max retries exceeded",
#             total_attempts=4
#         )
        
#         assert dlq_msg.original_message.message_id == "test-789"
#         assert dlq_msg.failure_reason == "Max retries exceeded"
#         assert dlq_msg.total_attempts == 4


# class TestSQSClient:
#     """SQS 클라이언트 테스트"""
    
#     @patch('app.queue.sqs_client.boto3.client')
#     def test_sqs_client_initialization(self, mock_boto_client):
#         """SQS 클라이언트 초기화 테스트"""
#         client = SQSClient(
#             queue_url="https://sqs.ap-northeast-2.amazonaws.com/123456/test-queue",
#             dlq_url="https://sqs.ap-northeast-2.amazonaws.com/123456/test-dlq",
#             region_name="ap-northeast-2",
#             aws_access_key_id="test-key",
#             aws_secret_access_key="test-secret"
#         )
        
#         assert client.queue_url == "https://sqs.ap-northeast-2.amazonaws.com/123456/test-queue"
#         mock_boto_client.assert_called_once()
    
#     @patch('app.queue.sqs_client.boto3.client')
#     def test_publish_message(self, mock_boto_client):
#         """메시지 발행 테스트"""
#         mock_sqs = MagicMock()
#         mock_sqs.send_message.return_value = {'MessageId': 'msg-12345'}
#         mock_boto_client.return_value = mock_sqs
        
#         client = SQSClient(
#             queue_url="https://sqs.test.queue",
#             region_name="ap-northeast-2",
#             aws_access_key_id="key",
#             aws_secret_access_key="secret"
#         )
        
#         msg = CallRequestMessage(
#             message_id="test-pub-1",
#             schedule_id=10,
#             elderly_id=100,
#             elderly_name="테스트",
#             phone_number="+821012345678",
#             scheduled_time=datetime.utcnow()
#         )
        
#         result = client.publish(msg)
        
#         assert result == "msg-12345"
#         mock_sqs.send_message.assert_called_once()
    
#     @patch('app.queue.sqs_client.boto3.client')
#     def test_receive_messages(self, mock_boto_client):
#         """메시지 수신 테스트"""
#         mock_sqs = MagicMock()
#         mock_sqs.receive_message.return_value = {
#             'Messages': [
#                 {'MessageId': 'msg-1', 'Body': '{}', 'ReceiptHandle': 'handle-1'},
#                 {'MessageId': 'msg-2', 'Body': '{}', 'ReceiptHandle': 'handle-2'}
#             ]
#         }
#         mock_boto_client.return_value = mock_sqs
        
#         client = SQSClient(
#             queue_url="https://sqs.test.queue",
#             region_name="ap-northeast-2",
#             aws_access_key_id="key",
#             aws_secret_access_key="secret"
#         )
        
#         messages = client.receive(max_messages=5)
        
#         assert len(messages) == 2
#         assert messages[0]['MessageId'] == 'msg-1'
    
#     @patch('app.queue.sqs_client.boto3.client')
#     def test_delete_message(self, mock_boto_client):
#         """메시지 삭제 테스트"""
#         mock_sqs = MagicMock()
#         mock_boto_client.return_value = mock_sqs
        
#         client = SQSClient(
#             queue_url="https://sqs.test.queue",
#             region_name="ap-northeast-2",
#             aws_access_key_id="key",
#             aws_secret_access_key="secret"
#         )
        
#         result = client.delete("receipt-handle-123")
        
#         assert result is True
#         mock_sqs.delete_message.assert_called_once()


# class TestSQSWorker:
#     """SQS 워커 테스트"""
    
#     @pytest.mark.asyncio
#     async def test_worker_process_message_success(self):
#         """메시지 처리 성공 테스트"""
#         mock_sqs = Mock(spec=SQSClient)
#         mock_sqs.delete.return_value = True
        
#         mock_callbot = Mock()
#         mock_callbot.make_call.return_value = None
        
#         worker = SQSWorker(
#             sqs_client=mock_sqs,
#             callbot_service=mock_callbot
#         )
        
#         msg = CallRequestMessage(
#             message_id="test-worker-1",
#             schedule_id=1,
#             elderly_id=100,
#             elderly_name="테스트노인",
#             phone_number="+821012345678",
#             scheduled_time=datetime.utcnow()
#         )
        
#         raw_message = {
#             'MessageId': 'test-worker-1',
#             'Body': msg.model_dump_json(),
#             'ReceiptHandle': 'handle-123'
#         }
        
#         result = await worker._process_message(raw_message)
        
#         assert result is True
#         mock_sqs.delete.assert_called_once_with('handle-123')
    
#     @pytest.mark.asyncio
#     async def test_worker_handle_failure_with_retry(self):
#         """실패 시 재시도 테스트"""
#         mock_sqs = Mock(spec=SQSClient)
#         mock_sqs.publish.return_value = "retry-msg-id"
#         mock_sqs.delete.return_value = True
        
#         worker = SQSWorker(sqs_client=mock_sqs)
        
#         msg = CallRequestMessage(
#             message_id="test-retry",
#             schedule_id=1,
#             elderly_id=100,
#             elderly_name="테스트",
#             phone_number="+821012345678",
#             scheduled_time=datetime.utcnow(),
#             retry_count=0  # 첫 시도
#         )
        
#         await worker._handle_failure(msg, "handle-456", "Test error")
        
#         # 재시도 메시지 발행 확인
#         mock_sqs.publish.assert_called_once()
#         mock_sqs.delete.assert_called_once()
    
#     @pytest.mark.asyncio
#     async def test_worker_handle_failure_max_retries(self):
#         """최대 재시도 초과 시 DLQ 이동 테스트"""
#         mock_sqs = Mock(spec=SQSClient)
#         mock_sqs.move_to_dlq.return_value = "dlq-msg-id"
#         mock_sqs.delete.return_value = True
        
#         worker = SQSWorker(sqs_client=mock_sqs)
        
#         msg = CallRequestMessage(
#             message_id="test-dlq",
#             schedule_id=1,
#             elderly_id=100,
#             elderly_name="테스트",
#             phone_number="+821012345678",
#             scheduled_time=datetime.utcnow(),
#             retry_count=3  # 최대 재시도 도달
#         )
        
#         await worker._handle_failure(msg, "handle-789", "Max retries test")
        
#         # DLQ 이동 확인
#         mock_sqs.move_to_dlq.assert_called_once()
#         mock_sqs.delete.assert_called_once()


# class TestDLQHandler:
#     """DLQ 핸들러 테스트"""
    
#     def test_dlq_handler_default_alert(self):
#         """기본 알림 처리 테스트"""
#         mock_sqs = Mock(spec=SQSClient)
#         mock_sqs.dlq_url = "https://sqs.test.dlq"
        
#         handler = DLQHandler(sqs_client=mock_sqs)
        
#         original_msg = CallRequestMessage(
#             message_id="test-alert",
#             schedule_id=1,
#             elderly_id=100,
#             elderly_name="알림테스트",
#             phone_number="+821012345678",
#             scheduled_time=datetime.utcnow()
#         )
        
#         dlq_msg = DLQMessage(
#             original_message=original_msg,
#             failure_reason="Test failure",
#             total_attempts=4
#         )
        
#         # 기본 알림은 로그 기록만 하므로 예외 없이 실행되면 성공
#         handler._default_alert(dlq_msg)
    
#     def test_dlq_handler_custom_alert(self):
#         """커스텀 알림 콜백 테스트"""
#         mock_sqs = Mock(spec=SQSClient)
#         alert_calls = []
        
#         def custom_alert(msg):
#             alert_calls.append(msg)
        
#         handler = DLQHandler(sqs_client=mock_sqs, alert_callback=custom_alert)
        
#         original_msg = CallRequestMessage(
#             message_id="test-custom",
#             schedule_id=2,
#             elderly_id=200,
#             elderly_name="커스텀테스트",
#             phone_number="+821098765432",
#             scheduled_time=datetime.utcnow()
#         )
        
#         dlq_msg = DLQMessage(
#             original_message=original_msg,
#             failure_reason="Custom test",
#             total_attempts=3
#         )
        
#         handler.process_dlq_messages([dlq_msg])
        
#         assert len(alert_calls) == 1
#         assert alert_calls[0].original_message.elderly_name == "커스텀테스트"
    
#     def test_reprocess_message(self):
#         """메시지 재처리 테스트"""
#         mock_sqs = Mock(spec=SQSClient)
#         mock_sqs.publish.return_value = "reprocessed-msg-id"
        
#         handler = DLQHandler(sqs_client=mock_sqs)
        
#         original_msg = CallRequestMessage(
#             message_id="test-reprocess",
#             schedule_id=5,
#             elderly_id=500,
#             elderly_name="재처리테스트",
#             phone_number="+821012345678",
#             scheduled_time=datetime.utcnow(),
#             retry_count=3
#         )
        
#         dlq_msg = DLQMessage(
#             original_message=original_msg,
#             failure_reason="Original failure",
#             total_attempts=4
#         )
        
#         result = handler.reprocess_message(dlq_msg)
        
#         assert result is True
#         mock_sqs.publish.assert_called_once()
        
#         # 재처리 시 retry_count가 0으로 리셋되었는지 확인
#         call_args = mock_sqs.publish.call_args
#         reprocessed_msg = call_args[0][0]
#         assert reprocessed_msg.retry_count == 0
