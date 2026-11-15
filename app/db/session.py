from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool
from app.config import settings

# Initialize the async engine with NullPool to prevent cross-loop thread crashes in LiveKit
engine = create_async_engine(
    settings.DATABASE_URL, 
    poolclass=NullPool, 
    echo=False
)

# Create a session factory
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

# Base class for our models
Base = declarative_base()