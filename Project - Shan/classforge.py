import sqlite3
import pandas as pd

def create_connection():
    """Create a database connection to the classforge.db SQLite database"""
    conn = None
    try:
        conn = sqlite3.connect('classforge.db')
        print("✅ Connected to SQLite database: classforge.db")
        return conn
    except sqlite3.Error as e:
        print(f"❌ Error connecting to database: {e}")
    return conn

def create_student_table(conn):
    """Create the students table"""
    try:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                academic_score REAL,
                wellbeing_score REAL,
                friendliness_score REAL,
                criticizes_score REAL,
                school_support_engage REAL
                -- Add more columns if your CSV has more fields
            )
        ''')
        conn.commit()
        print("✅ Table 'students' created successfully (or already exists)")
    except sqlite3.Error as e:
        print(f"❌ Error creating table: {e}")

def insert_students_from_csv(conn, csv_path):
    """Insert students from CSV into the students table"""
    try:
        df = pd.read_csv(csv_path)

        # Make sure the student_id column exists, if not create a unique one
        if 'student_id' not in df.columns:
            df.insert(0, 'student_id', ['student_' + str(i) for i in range(1, len(df)+1)])

        df.to_sql('students', conn, if_exists='replace', index=False)
        print(f"✅ Inserted {len(df)} rows into 'students' table from {csv_path}")
    except Exception as e:
        print(f"❌ Error inserting data from CSV: {e}")

def fetch_students(conn):
    """Fetch and display students"""
    try:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM students LIMIT 5')  # Fetch only a few rows to preview
        rows = cursor.fetchall()
        print("\n📋 Sample Data from 'students' table:")
        for row in rows:
            print(row)
    except sqlite3.Error as e:
        print(f"❌ Error fetching students: {e}")

def main():
    # Step 1: Connect to the database
    conn = create_connection()

    if conn is not None:
        # Step 2: Create the table
        create_student_table(conn)

        # Step 3: Insert students from CSV
        insert_students_from_csv(conn, 'synthetic_student_data.csv')

        # Step 4: Fetch and display some students
        fetch_students(conn)

        # Step 5: Close the connection
        conn.close()
        print("\n🔒 Connection closed.")
    else:
        print("❌ Cannot proceed without database connection.")

if __name__ == '__main__':
    main()
