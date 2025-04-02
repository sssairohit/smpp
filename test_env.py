from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables

print("TEST_VAR:", os.getenv("TEST_VAR"))
print("DEBUG:", os.getenv("DEBUG"))
