<div align="center">

# 🎙️ Voice-Order-Agent

### Multi-Mode Conversational Voice AI for Restaurants & Healthcare Clinics
Real-time Hindi / Hinglish telephony receptionist

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LiveKit Agents](https://img.shields.io/badge/LiveKit-WebRTC%20%2F%20SIP-00E5FF?style=for-the-badge&logo=webrtc&logoColor=black)](https://livekit.io/)
[![LangGraph](https://img.shields.io/badge/LangGraph-State%20Machine-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Supabase PostgreSQL](https://img.shields.io/badge/PostgreSQL-Transactional%20Outbox-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)](https://supabase.com/)
[![Deepgram Nova-2](https://img.shields.io/badge/STT-Deepgram%20Nova--2-13EF93?style=for-the-badge)](https://deepgram.com/)
[![n8n Automation](https://img.shields.io/badge/n8n-WhatsApp%20Webhooks-EA4B71?style=for-the-badge&logo=n8n&logoColor=white)](https://n8n.io/)

**Sub-second barge-in · Two-stage semantic relevance gate · Concurrency-safe booking · Atomic outbox dispatch**

</div>

---

## Table of Contents

1. [Overview](#overview)
2. [Key Engineering Highlights](#key-engineering-highlights)
3. [System Architecture](#system-architecture)
4. [Multi-Mode Domain Comparison](#multi-mode-domain-comparison)
5. [Project Structure](#project-structure)
6. [Database Schema & Concurrency Control](#database-schema--concurrency-control)
7. [Getting Started](#getting-started)
8. [Testing](#testing)
9. [Deployment](#deployment)

---

## Overview

**Voice-Order-Agent** is a production-oriented, full-duplex conversational voice AI system built for noisy Indian telephony environments. It combines **LiveKit WebRTC/SIP**, **Silero VAD**, **Deepgram Nova-2**, and **LangGraph** to hold natural **Hindi, Hinglish, and English** phone calls with instant barge-in support and deterministic database commits.

A Strategy / Router pattern lets a single codebase serve two verticals, selected with the `AGENT_MODE` environment variable:

| Mode | `AGENT_MODE` | Business | What it does |
| --- | --- | --- | --- |
| 🍛 Restaurant | `restaurant` | *Spice Villa* | Takes multi-item food orders, resolves fuzzy Hinglish menu aliases, verifies live database pricing, collects delivery details, and dispatches Kitchen Order Tickets (`#SV-...`). |
| 🦷 Dental Clinic | `clinic` | *Smile Dental Clinic* | Triages medical emergencies instantly, matches symptoms to treatments, resolves timezone-aware appointment slots under PostgreSQL advisory locks, and dispatches Patient Appointment Cards (`#APT-...`). |

---

## Key Engineering Highlights

### 1. Two-Stage Semantic Relevance Gate (Background Chatter Shield)

Phone calls are full of background TV audio, side conversations, and half-spoken interjections. Every transcript passes through a two-stage filter before it reaches the conversational state machine:

- **Stage 1 — Deterministic fast path (no LLM call):** Lexical, regex, and Devanagari-transliterated matching for domain keywords, emergency phrases, and mixed-script confirmations (e.g. `"हां confirm कर दो"`).
- **Stage 2 — LLM semantic filter + grounding check:** Classifies background chatter (`UNRELATED` / `AMBIGUOUS`) and verifies lexical grounding, so the agent never hallucinates items or resets the inactivity watchdog on background noise.

### 2. Acoustic Shielding & Instant Barge-In

- **Tuned Silero VAD:** 250 ms minimum speech threshold and 350 ms silence endpointing reject coughs, breaths, and speaker echo.
- **Zero-deadlock audio cancellation:** On `START_OF_SPEECH`, active TTS synthesis and FFmpeg audio pipes are terminated immediately and the outbound RTC audio queue is flushed.
- **Bounded inactivity and burst watchdogs:** Monotonic clocks (`InactivityClock`, `IdlePromptScheduler`, `GateFailurePrompter`) prevent zombie calls and recover gracefully from dropped network packets.

### 3. Race-Condition-Free Persistence & Transactional Outbox

- **PostgreSQL advisory locks (`pg_advisory_xact_lock`):** Serialize concurrent clinic bookings at the transaction layer so two simultaneous callers can never double-book the same slot.
- **Authoritative server-side pricing:** LLM arithmetic is never trusted. Totals are recomputed from `menu_items` / `CLINIC_SERVICES` inside an async SQLAlchemy transaction.
- **Transactional outbox:** `order_confirmed` and `appointment_confirmed` events are staged in `notification_outbox` within the **same ACID commit** as the booking, then dispatched asynchronously to **n8n** for WhatsApp ticket delivery.

---

## System Architecture

```mermaid
flowchart TD
    Caller(["📞 Caller (SIP / WebRTC)"]) <-->|"24kHz PCM audio"| LK["LiveKit Agent Worker"]

    subgraph VoicePipeline["Real-Time Audio & Guardrail Pipeline"]
        LK --> VAD["Silero VAD<br/>(barge-in & echo shield)"]
        LK --> STT["Deepgram Nova-2 STT<br/>(Hindi / Hinglish)"]
        VAD -->|"Interrupt"| TTSCancel["Cancel active TTS<br/>& flush queue"]
    end

    STT -->|"Final transcript"| Router{"AGENT_MODE<br/>router"}

    subgraph RestaurantEngine["🍛 Restaurant Engine (app/agent/)"]
        RGate["2-stage restaurant gate"] --> RGraph["LangGraph order<br/>state machine"]
        RGraph --> RMatch["Fuzzy menu &<br/>alias resolver"]
        RMatch --> RService["Order service<br/>(authoritative price check)"]
    end

    subgraph ClinicEngine["🦷 Clinic Engine (app/clinic/)"]
        CGate["2-stage clinic gate<br/>+ instant emergency triage"] --> CNodes["Clinic booking<br/>state machine"]
        CNodes --> CSlots["Timezone slot generator<br/>(Asia/Kolkata)"]
        CSlots --> CService["Appointment service<br/>(pg_advisory_xact_lock)"]
    end

    Router -->|"restaurant"| RGate
    Router -->|"clinic"| CGate

    RService --> DB[("Supabase PostgreSQL<br/>orders / appointments<br/>notification_outbox")]
    CService --> DB

    DB -->|"Async outbox batch"| Outbox["Outbox dispatcher"]
    Outbox -->|"Webhook POST"| N8N["n8n dual-mode workflow"]
    N8N -->|"Formatted alert"| WA(["💬 WhatsApp ticket / card"])

    RGraph -->|"Reply text"| TTS["Hinglish text normalizer<br/>+ streaming TTS"]
    CNodes -->|"Reply text"| TTS
    TTS -->|"24kHz s16le frames"| LK
```

---

## Multi-Mode Domain Comparison

| Feature | 🍛 Restaurant (`restaurant`) | 🦷 Dental Clinic (`clinic`) |
| --- | --- | --- |
| **Catalog source** | Supabase `menu_items` table | `app/clinic/catalog.py` + live DB slot check |
| **Primary entity** | Multi-item cart with quantities | Clinical treatment + ISO time slot (`slot_id`) |
| **Emergency triage** | N/A | Instant detection for bleeding, trauma, breathing issues |
| **State progression** | `ordering → awaiting_confirmation → collecting_details → closing` | `ordering → awaiting_confirmation → collecting_details → closing` |
| **Collected details** | Customer name + delivery address | Patient name + age / chief complaint |
| **Concurrency guard** | `idempotency_key` per call | `pg_advisory_xact_lock` + `SlotUnavailableError` mid-call recovery |
| **WhatsApp output** | 🛎️ SPICE VILLA — KITCHEN TICKET (`#SV-...`) | 🦷 SMILE DENTAL CLINIC — APPOINTMENT (`#APT-...`) |

---

## Project Structure

```text
voice-order-agent/
├── app/
│   ├── agent/                      # Restaurant voice engine & core worker
│   │   ├── graph.py                # LangGraph workflow compilation & menu loader
│   │   ├── livekit_worker.py       # Full-duplex WebRTC/SIP worker, VAD, mode router
│   │   ├── matching.py             # Phonetic & fuzzy Hinglish catalog matcher
│   │   ├── nodes.py                # Restaurant state machine & slot extractors
│   │   ├── relevance_gate.py       # 2-stage relevance gate & grounding verifier
│   │   └── state.py                # Typed session state schema
│   ├── clinic/                     # Isolated dental clinic booking engine
│   │   ├── catalog.py              # Treatments, fees, emergency phrases, IST slot generator
│   │   ├── gate.py                 # Clinic-specific relevance gate & fast-path rules
│   │   ├── nodes.py                # Mixed-script confirmation & appointment state machine
│   │   └── service.py              # Advisory-locked appointment persistence & slot loader
│   ├── db/
│   │   ├── models.py               # SQLAlchemy ORM models (Call, Order, OrderItem, MenuItem)
│   │   └── session.py              # AsyncEngine & AsyncSessionLocal factory
│   ├── services/
│   │   ├── order_service.py        # Atomic order + audit + outbox transaction
│   │   └── outbox_dispatcher.py    # Retrying webhook dispatcher with exponential backoff
│   └── voice/
│       └── speech_guard.py         # InactivityClock, IdlePromptScheduler, GateFailurePrompter
├── migrations/                     # Alembic database migrations
├── scripts/                        # Seeding, menu inspection, local testing utilities
├── tests/                          # Unit & state-transition test suites
├── Dockerfile                      # Production container for LiveKit Cloud
├── docker-compose.yml              # Local PostgreSQL & Redis services
├── livekit.toml                    # LiveKit Cloud agent configuration
└── requirements.txt                # Python dependencies
```

---

## Database Schema & Concurrency Control

Run the following in the Supabase SQL Editor to enable the clinic `appointments` table and the shared `notification_outbox`:

```sql
-- 1. Dedicated appointments table for Clinic Mode
CREATE TABLE IF NOT EXISTS public.appointments (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    appointment_number  VARCHAR(32) UNIQUE NOT NULL,
    clinic_id           UUID NOT NULL,
    call_id             UUID REFERENCES public.calls(id),
    call_sid            VARCHAR(128),
    patient_name        VARCHAR(128) NOT NULL,
    patient_phone       VARCHAR(32) DEFAULT 'UNAVAILABLE',
    patient_age_or_note VARCHAR(255),
    service_name        VARCHAR(128) NOT NULL,
    appointment_slot    VARCHAR(128) NOT NULL,
    slot_start          TIMESTAMPTZ,
    consultation_fee    NUMERIC(10, 2) NOT NULL DEFAULT 300.00,
    status              VARCHAR(32) NOT NULL DEFAULT 'CONFIRMED',
    idempotency_key     VARCHAR(128) UNIQUE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 2. Indexes for idempotency & fast slot-capacity checks
CREATE UNIQUE INDEX IF NOT EXISTS uq_appointments_idempotency_key
    ON public.appointments (idempotency_key);

CREATE INDEX IF NOT EXISTS ix_appointments_clinic_slot
    ON public.appointments (clinic_id, slot_start)
    WHERE status = 'CONFIRMED';

-- 3. Let notification_outbox hold both restaurant orders and clinic appointments
ALTER TABLE public.notification_outbox
    DROP CONSTRAINT IF EXISTS notification_outbox_order_id_fkey;
```

---

## Getting Started

### 1. Prerequisites

- Python 3.11+ and FFmpeg available on your `PATH`
- A LiveKit Cloud project and the [`lk` CLI](https://docs.livekit.io/home/cli/)
- A Supabase PostgreSQL database
- A Groq API key (LLM structured extraction) and a Deepgram API key (Nova-2 Hindi STT)
- n8n (for WhatsApp webhook automation)

### 2. Installation

```bash
git clone https://github.com/technospes/voice-order-agent.git
cd voice-order-agent

python -m venv venv

# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Environment Configuration

Copy `.env.example` to `.env` and fill in your credentials:

```env
# Mode selection: "restaurant" or "clinic"
AGENT_MODE=clinic
AGENT_NAME=order-agent

# LiveKit Cloud
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_livekit_key
LIVEKIT_API_SECRET=your_livekit_secret

# AI providers
GROQ_API_KEY=your_groq_api_key
DEEPGRAM_API_KEY=your_deepgram_api_key

# Database (Supabase asyncpg pooler / direct URL)
DATABASE_URL=postgresql+asyncpg://postgres:password@db.your-project.supabase.co:5432/postgres

# Clinic configuration
CLINIC_TZ=Asia/Kolkata
CLINIC_MIN_LEAD_MIN=30
CLINIC_SLOT_CAPACITY=1

# Outbox webhook (n8n)
N8N_ORDER_WEBHOOK_URL=https://your-ngrok-or-n8n-domain/webhook/orders/new
```

---

## Testing

Run the automated suite covering the relevance gate, lexical grounding, confirmation classifiers, and state transitions:

```bash
pytest tests/ -v
```

Verify mixed-script Hindi/Hinglish confirmation handling from the CLI:

```bash
python -c "from app.clinic.nodes import classify_clinic_confirmation; print(classify_clinic_confirmation('हां confirm कर दो.')); print(classify_clinic_confirmation('कर दो confirm?'))"
```

---

## Deployment

Clear local bytecode caches, verify compilation, and deploy the worker to LiveKit Cloud (Windows `cmd`):

```bat
for /d /r . %d in (__pycache__) do @if exist "%d" rd /s /q "%d"
python -m compileall app
lk agent deploy --secrets-file .env
```

Monitor live call telemetry and state transitions:

```bash
lk agent logs
```