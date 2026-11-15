import asyncio
from sqlalchemy import text
from app.db.session import AsyncSessionLocal

async def reset_menu_tables():
    async with AsyncSessionLocal() as session:
        # CASCADE ensures dependent menu_items are also deleted
        await session.execute(text("TRUNCATE TABLE menu_categories CASCADE;"))
        await session.commit()
        print("✅ Menu tables cleared successfully!")

if __name__ == "__main__":
    asyncio.run(reset_menu_tables())