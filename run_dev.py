import os
import sys
import traceback
import contextlib
from main import app
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

@contextlib.contextmanager
def suppress_stderr():
    """Temporarily suppress stderr (e.g., psycopg2 native tracebacks)"""
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

def main():
    print("🧪 Running dev server...")

    # ✅ Suppress native tracebacks during DB check
    with suppress_stderr():
        try:
            engine = create_engine(os.environ.get("DB_URI"))
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("✅ DB check passed")
        except OperationalError as e:
            print("❌ Cannot connect to DB:")
            print(str(e))  # only show clean error
            return
        except Exception:
            traceback.print_exc()
            return

    # ✅ Launch Flask app if DB is okay
    app.run(debug=True)

if __name__ == "__main__":
    main()
