from typing import Any
from fastapi import Request, Response, HTTPException
from twilio.request_validator import RequestValidator
from app.config import settings
from app.telephony.base import IncomingCall

class TwilioAdapter:
    def __init__(self):
        self.validator = RequestValidator(settings.TWILIO_AUTH_TOKEN)

    async def verify_request(self, request: Request) -> bool:
        form_data = await request.form()
        signature = request.headers.get("X-Twilio-Signature", "")
        url = str(request.url)
        
        # Twilio proxy URLs differ from ngrok URLs; enforce https protocol matching
        url = url.replace("http://", "https://") if "ngrok" in url else url
        
        if not self.validator.validate(url, dict(form_data), signature):
            raise HTTPException(status_code=403, detail="Invalid Twilio signature")
        return True

    def parse_incoming(self, form_data: dict) -> IncomingCall:
        return IncomingCall(
            call_sid=form_data.get("CallSid", ""),
            from_number=form_data.get("From", ""),
            to_number=form_data.get("To", "")
        )

    def answer_with_stream(self, call: IncomingCall, wss_url: str) -> Response:
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
        <Response>
            <Connect>
                <Stream url="{wss_url}" />
            </Connect>
        </Response>"""
        return Response(content=twiml, media_type="application/xml")
        
    def reject_call(self) -> Response:
        twiml = """<?xml version="1.0" encoding="UTF-8"?><Response><Reject /></Response>"""
        return Response(content=twiml, media_type="application/xml")