import asyncio
from sqlalchemy.future import select
from app.db.session import AsyncSessionLocal
from app.db.models import Restaurant, MenuItem

async def check_db():
    async with AsyncSessionLocal() as session:
        # Get the first restaurant
        res = await session.execute(select(Restaurant))
        restaurant = res.scalars().first()
        
        if not restaurant:
            print("❌ No restaurant found in the database.")
            return
            
        print(f"✅ Restaurant Found: {restaurant.name} | ID: {restaurant.id}")
        
        # Get menu items for this restaurant
        res = await session.execute(select(MenuItem).where(MenuItem.restaurant_id == restaurant.id))
        items = res.scalars().all()
        
        print("\n📋 Menu Items in Database:")
        if not items:
            print("   No items found.")
        for item in items:
            # Changed from item.price to item.base_price
            print(f"   - {item.name} (₹{item.base_price})")

if __name__ == "__main__":
    asyncio.run(check_db())