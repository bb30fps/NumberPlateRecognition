import sqlite3
import datetime

class DatabaseManager:
    def __init__(self):
        self.conn = sqlite3.connect('project.db')
        self._create_tables()
    
    def _create_tables(self):
        c = self.conn.cursor()
        # Users table
        c.execute('''CREATE TABLE IF NOT EXISTS users 
                     (id INTEGER PRIMARY KEY,
                      username TEXT UNIQUE,
                      password TEXT,
                      is_admin BOOLEAN)''')
        # Logs table
        c.execute('''CREATE TABLE IF NOT EXISTS logs 
                     (id INTEGER PRIMARY KEY,
                      timestamp DATETIME,
                      user TEXT,
                      action TEXT)''')
        self.conn.commit()
    
    def authenticate_user(self, username, password):
        c = self.conn.cursor()
        c.execute("SELECT is_admin FROM users WHERE username=? AND password=?", 
                 (username, password))
        return c.fetchone()
    
    def log_action(self, username, action):
        c = self.conn.cursor()
        c.execute("INSERT INTO logs (timestamp, user, action) VALUES (?, ?, ?)",
                 (datetime.datetime.now(), username, action))
        self.conn.commit()
    
    def add_user(self, username, password, is_admin=False):
        try:
            c = self.conn.cursor()
            c.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                     (username, password, is_admin))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
