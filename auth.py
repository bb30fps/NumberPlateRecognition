class AuthManager:
    def __init__(self, db):
        self.db = db
    
    def validate_permissions(self, username, required_level):
        """Check user permissions against database"""
        cursor = self.db.conn.cursor()
        
        if required_level == "admin":
            cursor.execute(
                "SELECT is_admin FROM users WHERE username=?",
                (username,)
            )
            result = cursor.fetchone()
            return result[0] if result else False
            
        return True  # Default allow for non-admin actions
