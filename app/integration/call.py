from twilio.rest import Client
import urllib.parse

class CALL:
    def __init__(self, account_sid: str, auth_token: str, url: str, number: str, silverlink_number: str):
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.url = url
        self.number = number
        self.silverlink_number = silverlink_number
        
    def calling(self, elderly_id: int, phone_number: str, elderly_name: str, initial_mem: str = "", greeting: str = "") -> None:
        # 계정 정보
        account_sid = self.account_sid
        auth_token = self.auth_token
        client = Client(account_sid, auth_token)

        # 데이터를 URL 쿼리 파라미터로 추가
        params = {
            "elderly_id": elderly_id,
            "elderly_name": elderly_name,
            "initial_mem": initial_mem,
            "greeting": greeting  # 미리 생성된 인삿말 추가
        }
        query_string = urllib.parse.urlencode(params)
        my_server_url = f"{self.url}/api/callbot/voice?{query_string}"

        call = client.calls.create(
            to=phone_number,      # 받는 사람 번호
            from_=self.silverlink_number,    # Twilio 발신 번호
            url=my_server_url,        # 우리가 만든 AI 서버 주소
            status_callback=f"{self.url}/api/callbot/status",
            status_callback_event=["completed"],
            record=True,
            recording_status_callback=f"{self.url}/api/callbot/s3-upload",
            recording_status_callback_method='POST'
        )
        
        # Twilio call 객체 반환 (call.sid 포함)
        return {
            "call_sid": call.sid,
            "status": call.status,
            "to": call.to,
            "from": call.from_formatted  # from_ 대신 from_formatted 사용
        }
        
