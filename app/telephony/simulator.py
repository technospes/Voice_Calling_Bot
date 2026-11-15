from typing import Any, Dict, Optional
from app.telephony.base import IncomingCall

class SimulatorAdapter:
    def verify_request(self, request: Any) -> bool:
        return True

    def parse_incoming(self, request: Dict[str, Any]) -> IncomingCall:
        # Defaults to the seed data numbers if not provided
        return IncomingCall(
            call_sid=request.get("call_sid", "sim_12345"),
            from_number=request.get("from_number", "+918888888888"),
            to_number=request.get("to_number", "+910000000000")
        )

    def answer_with_stream(self, call: IncomingCall, wss_url: str) -> Dict[str, Any]:
        return {"action": "answer", "stream_url": wss_url}

    def parse_dtmf(self, payload: Dict[str, Any]) -> Optional[str]:
        return payload.get("digits")

    def forward_call(self, call: IncomingCall, to_number: str) -> Dict[str, Any]:
        return {"action": "forward", "to": to_number}