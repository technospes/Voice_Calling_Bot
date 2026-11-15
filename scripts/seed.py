import asyncio
import sys
import os

# Ensure the 'app' module can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.future import select
from app.db.session import AsyncSessionLocal
from app.db.models import Restaurant, StaffContact
from app.db.models import MenuCategory, MenuItem

async def seed():
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Restaurant).where(Restaurant.slug == "spice-villa"))
        restaurant = result.scalars().first()
        
        if not restaurant:
            restaurant = Restaurant(
                name="Spice Villa",
                slug="spice-villa",
                inbound_number="+910000000000",  # We will replace this with your Twilio/provider number later
                owner_forward_number="+919999999999",  # Replace with the mobile number that will test "Press 3 for Owner"
                greeting_text="Spice Villa mein aapka swagat hai. Order dene ke liye 1 dabaiye, menu sunne ke liye 2, owner se baat karne ke liye 3."
            )
            session.add(restaurant)
            await session.flush()  # Retrieve the generated UUID

            owner = StaffContact(
                restaurant_id=restaurant.id,
                name="Test Owner",
                role="OWNER",
                whatsapp_number="+919999999999"
            )
            session.add(owner)

        # Check if category exists, if not create it
        cat_result = await session.execute(select(MenuCategory).where(MenuCategory.restaurant_id == restaurant.id))
        category = cat_result.scalars().first()
        if not category:
            category = MenuCategory(restaurant_id=restaurant.id, name="Main Course")
            session.add(category)
            await session.flush()
            
        # Define the desired menu items including the new Naan varieties
        desired_items = [
            {"name": "Paneer Tikka", "aliases": ["paneer tikka", "panir tika"], "price": 260.00},
            {"name": "Butter Naan", "aliases": ["butter naan", "nan", "makhni naan"], "price": 50.00},
            {"name": "Paneer Butter Masala", "aliases": ["paneer butter masala", "butter paneer"], "price": 300.00},
            {"name": "Garlic Naan", "aliases": ["garlic naan", "lahsuni naan", "lasooni naan"], "price": 60.00},
            {"name": "Plain Naan", "aliases": ["plain naan", "sada naan"], "price": 40.00},
        ]

        # Check existing items to avoid duplicates
        item_result = await session.execute(select(MenuItem).where(MenuItem.restaurant_id == restaurant.id))
        existing_items = {item.name for item in item_result.scalars().all()}

        new_items = []
        for data in desired_items:
            if data["name"] not in existing_items:
                new_items.append(
                    MenuItem(
                        restaurant_id=restaurant.id, 
                        category_id=category.id, 
                        name=data["name"], 
                        aliases=data["aliases"], 
                        base_price=data["price"]
                    )
                )

        if new_items:
            session.add_all(new_items)
            await session.commit()
            print(f"Added {len(new_items)} new menu items to the database!")
        else:
            print("Menu items are already up to date.")

        print(f"Successfully seeded restaurant: {restaurant.name} (ID: {restaurant.id})")

if __name__ == "__main__":
    asyncio.run(seed())