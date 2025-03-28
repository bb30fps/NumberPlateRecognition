class AuthManager:
    def __init__(self, db):
        self.db = db
    
    def validate_permissions(self, user, required_level):
        if required_level == "admin":
            return self.db.is_admin(user)
        return True
