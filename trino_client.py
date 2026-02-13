import os
from dotenv import load_dotenv
import trino
from trino.auth import OAuth2Authentication

load_dotenv()

OPENSKY_USER = os.getenv("OPENSKY_USER")
if not OPENSKY_USER:
    raise RuntimeError("OPENSKY_USER not set in .env")


def get_trino_connection():
    return trino.dbapi.connect(
        host="trino.opensky-network.org",
        port=443,
        http_scheme="https",
        user=OPENSKY_USER,
        auth=OAuth2Authentication(),
        catalog="minio",
        schema="osky",
    )
