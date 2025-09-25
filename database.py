import sqlite3
from datetime import datetime
import pandas as pd
import json

class BookStoreDB:
    def __init__(self, db_path="data/bookstore.db"):
        self.db_path = db_path
        self.init_database()
        self.seed_data()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tạo bảng Books
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS books (
                book_id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                author TEXT,
                price REAL NOT NULL,
                stock INTEGER NOT NULL,
                category TEXT NOT NULL)
                ''')
        
        # Tạo bảng Orders
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                order_id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_name TEXT NOT NULL,
                phone TEXT,
                address TEXT,
                book_id INTEGER NOT NULL,
                quantity INTEGER NOT NULL,
                total_price REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (book_id) REFERENCES books (book_id)
            )
        ''')
        
        conn.commit()
        conn.close()

    def seed_data(self):
        df = pd.read_csv("data/books.csv")

        df['original_price'] = df['original_price'] * 24000
        df = df[['product_id', 'title', 'authors', 'original_price', 'quantity', 'category']]

        df = df.rename(columns={
            "product_id": "book_id",
            "title": "title",
            "authors": "author",
            "original_price": "price",
            "quantity": "stock",
            "category": "category"
        })

        # 3. Kết nối database
        conn = sqlite3.connect(self.db_path)

        # 4. Insert vào bảng books
        conn.cursor().execute("DELETE FROM books")
        conn.commit()
        df.to_sql("books", conn, if_exists="replace", index=False)


        

        conn.commit()
        conn.close()
