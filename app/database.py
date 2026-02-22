from pathlib import Path
from datetime import date
import sqlite3

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "app.db"

DB_PATH.parent.mkdir(exist_ok=True)

conn = sqlite3.connect(DB_PATH)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS rent_payments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_date TEXT NOT NULL,     -- 2026-02-01（家賃対象月）
    amount INTEGER NOT NULL,
    paid_date TEXT,
    status TEXT CHECK(status IN ('paid','unpaid')) DEFAULT 'unpaid',
    note TEXT
);
""")




def create_data():
    start_year = 2020
    start_month = 1
    end_year = 2026
    end_month = 1

    year = start_year
    month = start_month

    while (year < end_year) or (year == end_year and month <= end_month):

        target_date = date(year, month, 1).isoformat()
        paid_date = date(year, month, 26).isoformat()

        cursor.execute("""
            INSERT INTO rent_payments
            (target_date, amount, paid_date, status, note)
            VALUES (?, ?, ?, ?, ?)
        """, (target_date, 60000, paid_date, "paid", "自動登録"))

    # 月を進める
        month += 1
        if month > 12:
            month = 1
            year += 1

    #conn.commit()
    #conn.close()

    print("登録完了")

def show_all_data():
#    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM rent_payments")
        rows = cursor.fetchall()

        for row in rows:
            print(row)


if __name__ == "__main__":
    #create_data()
    show_all_data()

conn.commit()
conn.close()
