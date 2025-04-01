class AuthManager:
    def __init__(self, db):
        self.db = db
    
    def validate_permissions(self, username, required_level):
        """Checks if user has required access level"""
        if required_level == "admin":
            # Get admin status from users table
            c = self.db.conn.cursor()
            c.execute("SELECT is_admin FROM users WHERE username=?", (username,))
            result = c.fetchone()
            return result[0] if result else False
        return True
