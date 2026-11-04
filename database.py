
import sqlite3
import uuid

DATABASE_NAME = "chat_history.db"


# 1. Database connection
def get_connection():
    connection = sqlite3.connect(DATABASE_NAME)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


# 2. Create database tables
def create_tables():
    connection = get_connection()
    cursor = connection.cursor()

    # Conversations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            document_id TEXT
        )
    """)

    # Messages table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id)
                REFERENCES conversations(id)
                ON DELETE CASCADE
        )
    """)

    # Check existing conversation columns
    cursor.execute("PRAGMA table_info(conversations)")
    conversation_columns = [
        row["name"] for row in cursor.fetchall()
    ]

    # Add document_id if it does not exist
    if "document_id" not in conversation_columns:
        cursor.execute("""
            ALTER TABLE conversations
            ADD COLUMN document_id TEXT
        """)

    connection.commit()
    connection.close()


# 3. Create a new conversation
def create_conversation(title="New Chat"):
    connection = get_connection()

    conversation_id = str(uuid.uuid4())

    connection.execute(
        """
        INSERT INTO conversations (id, title)
        VALUES (?, ?)
        """,
        (conversation_id, title)
    )

    connection.commit()
    connection.close()

    return conversation_id


# 4. Save a message
def save_message(conversation_id, role, content):
    connection = get_connection()

    cursor = connection.execute(
        """
        INSERT INTO messages
        (conversation_id, role, content)
        VALUES (?, ?, ?)
        """,
        (conversation_id, role, content)
    )

    message_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return message_id


# 5. Get messages for a conversation
def get_messages(conversation_id):
    connection = get_connection()

    cursor = connection.execute(
        """
        SELECT
            id,
            conversation_id,
            role,
            content,
            created_at
        FROM messages
        WHERE conversation_id = ?
        ORDER BY created_at ASC, rowid ASC
        """,
        (conversation_id,)
    )

    messages = [
        dict(row) for row in cursor.fetchall()
    ]

    connection.close()

    return messages


# 6. Get all conversations
def get_conversations():
    connection = get_connection()

    cursor = connection.execute(
        """
        SELECT
            c.id,
            c.title,
            c.document_id,
            c.created_at,
            (
                SELECT content
                FROM messages
                WHERE conversation_id = c.id
                ORDER BY created_at DESC, rowid DESC
                LIMIT 1
            ) AS last_message

        FROM conversations c

        ORDER BY c.created_at DESC
        """
    )

    conversations = [
        dict(row) for row in cursor.fetchall()
    ]

    connection.close()

    return conversations


# 7. Get one conversation
def get_conversation(conversation_id):
    connection = get_connection()

    cursor = connection.execute(
        """
        SELECT *
        FROM conversations
        WHERE id = ?
        """,
        (conversation_id,)
    )

    conversation = cursor.fetchone()

    connection.close()

    return dict(conversation) if conversation else None


# 8. Associate PDF with a conversation
def set_conversation_document(conversation_id, document_id):
    connection = get_connection()

    cursor = connection.execute(
        """
        UPDATE conversations
        SET document_id = ?
        WHERE id = ?
        """,
        (document_id, conversation_id)
    )

    connection.commit()

    updated = cursor.rowcount > 0

    connection.close()

    return updated


# 9. Delete a conversation
def delete_conversation(conversation_id):
    connection = get_connection()

    cursor = connection.execute(
        """
        DELETE FROM conversations
        WHERE id = ?
        """,
        (conversation_id,)
    )

    connection.commit()

    deleted = cursor.rowcount > 0

    connection.close()

    return deleted
