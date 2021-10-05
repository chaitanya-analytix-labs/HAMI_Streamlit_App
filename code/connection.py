import sqlite3
import hashlib


class Connections:
    conn = sqlite3.connect('data.db', check_same_thread=False)
    c = conn.cursor()

    def create_usertable(self):
        self.c.execute('CREATE TABLE IF NOT EXISTS usertable(username TEXT,password TEXT,email TEXT)')

    def add_userdata(self, username, password):
        self.c.execute('INSERT INTO usertable(username,password) VALUES (?,?)', (username, password))
        self.conn.commit()

    def login_user(self, username, password):
        self.c.execute('SELECT * from usertable WHERE username = ? AND password = ?', (username, password))
        data = self.c.fetchall()
        return data

    def view_all_users(self):
        self.c.execute('SELECT * FROM usertable')
        data = self.c.fetchall()
        return data


# sql database management to store passwords

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False
