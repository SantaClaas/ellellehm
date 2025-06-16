from sqlite3 import Connection
import sqlite3


def initialize() -> Connection:
    connection = sqlite3.connect(".db")

    # Not storing anything that is not required based on GDPR data minimalism
    connection.execute("""
CREATE TABLE IF NOT EXISTS token_usages (input_tokens INTEGER, output_tokens INTEGER);
""")

    return connection


def create_usage(connection: Connection, input_tokens: int, output_tokens: int):
    connection.execute(
        """INSERT INTO token_usages (input_tokens, output_tokens) VALUES (?, ?)""", (input_tokens, output_tokens))
    connection.commit()
