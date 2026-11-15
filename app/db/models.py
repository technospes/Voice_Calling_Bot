import uuid
import enum
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Boolean, DateTime, Enum, ForeignKey, Numeric, Float, ARRAY, JSON, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from app.db.session import Base

class CallStatus(str, enum.Enum):
    RINGING = "RINGING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    DROPPED = "DROPPED"
    TRANSFERRED = "TRANSFERRED"
    FAILED = "FAILED"

class Speaker(str, enum.Enum):
    BOT = "BOT"
    CUSTOMER = "CUSTOMER"

def utcnow():
    return datetime.now(timezone.utc)

class Restaurant(Base):
    __tablename__ = "restaurants"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    slug = Column(String, unique=True, nullable=False)
    inbound_number = Column(String, unique=True, nullable=False)
    owner_forward_number = Column(String, nullable=False)
    default_language = Column(String, nullable=False, default="hi-IN")
    greeting_text = Column(String)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

class StaffContact(Base):
    __tablename__ = "staff_contacts"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    restaurant_id = Column(UUID(as_uuid=True), ForeignKey("restaurants.id", ondelete="CASCADE"), nullable=False)
    name = Column(String, nullable=False)
    role = Column(String, nullable=False) # 'OWNER', 'MANAGER', 'KITCHEN'
    whatsapp_number = Column(String, nullable=False)
    receives_orders = Column(Boolean, nullable=False, default=True)

class Call(Base):
    __tablename__ = "calls"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    restaurant_id = Column(UUID(as_uuid=True), ForeignKey("restaurants.id"), nullable=False)
    provider_call_sid = Column(String, unique=True, nullable=False)
    from_number = Column(String, nullable=False)
    to_number = Column(String, nullable=False)
    status = Column(Enum(CallStatus), nullable=False, default=CallStatus.RINGING)
    started_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    duration_sec = Column(Integer)
    recording_url = Column(String)
    forwarded_to_owner = Column(Boolean, nullable=False, default=False)

class CallTurn(Base):
    __tablename__ = "call_turns"
    id = Column(Integer, primary_key=True, autoincrement=True)
    call_id = Column(UUID(as_uuid=True), ForeignKey("calls.id", ondelete="CASCADE"), nullable=False)
    turn_no = Column(Integer, nullable=False)
    speaker = Column(Enum(Speaker), nullable=False)
    text = Column(String)
    dtmf = Column(String)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

class OrderStatus(str, enum.Enum):
    DRAFT = "DRAFT"
    CONFIRMED = "CONFIRMED"
    PREPARING = "PREPARING"
    OUT_FOR_DELIVERY = "OUT_FOR_DELIVERY"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    ABANDONED = "ABANDONED"

class OrderType(str, enum.Enum):
    DELIVERY = "DELIVERY"
    PICKUP = "PICKUP"

class PaymentMode(str, enum.Enum):
    COD = "COD"
    UPI_ON_DELIVERY = "UPI_ON_DELIVERY"

class MenuCategory(Base):
    __tablename__ = "menu_categories"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    restaurant_id = Column(UUID(as_uuid=True), ForeignKey("restaurants.id", ondelete="CASCADE"), nullable=False)
    name = Column(String, nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)

class MenuItem(Base):
    __tablename__ = "menu_items"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    restaurant_id = Column(UUID(as_uuid=True), ForeignKey("restaurants.id", ondelete="CASCADE"), nullable=False)
    category_id = Column(UUID(as_uuid=True), ForeignKey("menu_categories.id", ondelete="SET NULL"))
    name = Column(String, nullable=False)
    aliases = Column(ARRAY(String), nullable=False, default=list)
    base_price = Column(Numeric(10, 2), nullable=False)
    is_available = Column(Boolean, nullable=False, default=True)

class Order(Base):
    __tablename__ = "orders"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    order_number = Column(String, unique=True, nullable=False)
    restaurant_id = Column(UUID(as_uuid=True), ForeignKey("restaurants.id"), nullable=False)
    call_id = Column(UUID(as_uuid=True), ForeignKey("calls.id"), unique=True)
    
    # Use Enum(OrderStatus) with values_callable to match PostgreSQL's enum type
    status = Column(Enum(OrderStatus, name="orderstatus", create_type=False), default=OrderStatus.CONFIRMED, nullable=False)
    
    idempotency_key = Column(String(128), unique=True, nullable=True)
    call_sid = Column(String(128), nullable=True)
    staff_notes = Column(Text, nullable=True)
    order_type = Column(Enum(OrderType), nullable=False, default=OrderType.DELIVERY)
    customer_phone = Column(String, nullable=False)
    delivery_address = Column(String)
    total_amount = Column(Numeric(10, 2), nullable=False, default=0)
    payment_mode = Column(Enum(PaymentMode), nullable=False, default=PaymentMode.COD)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

class OrderItem(Base):
    __tablename__ = "order_items"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    order_id = Column(UUID(as_uuid=True), ForeignKey("orders.id", ondelete="CASCADE"), nullable=False)
    item_id = Column(UUID(as_uuid=True), ForeignKey("menu_items.id", ondelete="SET NULL"))
    item_name = Column(String, nullable=False)
    quantity = Column(Integer, nullable=False)
    unit_price = Column(Numeric(10, 2), nullable=False)
    line_total = Column(Numeric(10, 2), nullable=False)

class MsgDirection(str, enum.Enum):
    OUTBOUND = "OUTBOUND"
    INBOUND = "INBOUND"

class MsgStatus(str, enum.Enum):
    QUEUED = "QUEUED"
    SENT = "SENT"
    DELIVERED = "DELIVERED"
    READ = "READ"
    FAILED = "FAILED"

class WhatsAppMessage(Base):
    __tablename__ = "whatsapp_messages"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    order_id = Column(UUID(as_uuid=True), ForeignKey("orders.id", ondelete="SET NULL"), nullable=True)
    restaurant_id = Column(UUID(as_uuid=True), ForeignKey("restaurants.id", ondelete="CASCADE"), nullable=False)
    direction = Column(Enum(MsgDirection), nullable=False)
    recipient_role = Column(String, nullable=False)
    to_number = Column(String, nullable=False)
    template_name = Column(String, nullable=True)
    provider_message_id = Column(String, nullable=True)
    error_message = Column(String, nullable=True)
    status = Column(Enum(MsgStatus), nullable=False, default=MsgStatus.QUEUED)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

class NotificationOutbox(Base):
    __tablename__ = "notification_outbox"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    order_id = Column(UUID(as_uuid=True), ForeignKey("orders.id"), nullable=True)
    event_type = Column(String(50), nullable=False, default="order_created")
    payload = Column(JSONB, nullable=False)
    status = Column(String(20), nullable=False, default="pending")  # pending, processing, sent, failed, dead_letter
    attempts = Column(Integer, nullable=False, default=0)
    last_error = Column(Text, nullable=True)
    next_retry_at = Column(DateTime(timezone=True), nullable=True)
    sent_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

__all__ = [
    "CallStatus",
    "Speaker",
    "Restaurant",
    "StaffContact",
    "Call",
    "CallTurn",
    "OrderStatus",
    "OrderType",
    "PaymentMode",
    "MenuCategory",
    "MenuItem",
    "Order",
    "OrderItem",
    "MsgDirection",
    "MsgStatus",
    "WhatsAppMessage",
    "NotificationOutbox",
]