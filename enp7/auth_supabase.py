from supabase import create_client, Client
import os

# Supabase credentials
SUPABASE_URL = "https://kyanyydvgfdjsirsjmvr.supabase.co"
SUPABASE_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imt5YW55eWR2Z2ZkanNpcnNqbXZyIiwicm9sZSI6ImFub24i"
    "LCJpYXQiOjE3NTAyNTg4MzgsImV4cCI6MjA2NTgzNDgzOH0."
    "GkZXkMoZgP5hx8sz7fh4LVaXIa-MHzbwJPup_ZKSiTs"
)

# Create Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def register_user(email: str, password: str, role: str = "user") -> tuple:
    try:
        # Sign up user
        response = supabase.auth.sign_up({"email": email, "password": password})
        user = response.user

        # Insert into profile table
        if user:
            supabase.table("profiles").insert({
                "id": user.id,
                "email": email,
                "role": role
            }).execute()

        return True, "User registered successfully!"
    except Exception as e:
        return False, str(e)

def login_user(email: str, password: str) -> tuple:
    try:
        response = supabase.auth.sign_in_with_password({"email": email, "password": password})
        return response.session, response.user
    except Exception as e:
        return None, None

def get_user_role(user_id: str) -> str:
    try:
        response = (
            supabase.table("profiles")
            .select("role")
            .eq("id", user_id)
            .single()
            .execute()
        )
        return response.data.get("role", "user")
    except Exception:
        return "user"

def logout_user() -> None:
    try:
        supabase.auth.sign_out()
    except Exception:
        pass
