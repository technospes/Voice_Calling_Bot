from typing import Protocol, Any, Dict, Optional

class IncomingCall:
    def __init__(self, call_sid: str, from_number: str, to_number: str):
        self.call_sid = call_sid
        self.from_number = from_number
        self.to_number = to_number

class TelephonyAdapter(Protocol):
    def verify_request(self, request: Any) -> bool:
        """Verify the webhook signature or IP address."""
        ...

    def parse_incoming(self, request: Any) -> IncomingCall:
        """Extract call metadata from the provider's incoming webhook payload."""
        ...

    def answer_with_stream(self, call: IncomingCall, wss_url: str) -> Any:
        """Return the provider-specific response to answer the call and open a WebSocket."""
        ...

    def parse_dtmf(self, payload: Any) -> Optional[str]:
        """Extract the DTMF digit if sent via webhook."""
        ...

    def forward_call(self, call: IncomingCall, to_number: str) -> Any:
        """Return the provider-specific response to transfer the call to a human."""
        ...