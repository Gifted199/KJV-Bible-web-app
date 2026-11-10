"""The 66 books of the KJV in order, with their testament and traditional group."""

BOOK_GROUPS = {
    "Law": ["Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy"],
    "History": [
        "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel", "1 Kings", "2 Kings",
        "1 Chronicles", "2 Chronicles", "Ezra", "Nehemiah", "Esther",
    ],
    "Poetry and Wisdom": ["Job", "Psalms", "Proverbs", "Ecclesiastes", "Song of Solomon"],
    "Major Prophets": ["Isaiah", "Jeremiah", "Lamentations", "Ezekiel", "Daniel"],
    "Minor Prophets": [
        "Hosea", "Joel", "Amos", "Obadiah", "Jonah", "Micah", "Nahum",
        "Habakkuk", "Zephaniah", "Haggai", "Zechariah", "Malachi",
    ],
    "Gospels": ["Matthew", "Mark", "Luke", "John"],
    "Church History": ["Acts"],
    "Pauline Epistles": [
        "Romans", "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians", "Philippians",
        "Colossians", "1 Thessalonians", "2 Thessalonians", "1 Timothy", "2 Timothy", "Titus", "Philemon",
    ],
    "General Epistles": ["Hebrews", "James", "1 Peter", "2 Peter", "1 John", "2 John", "3 John", "Jude"],
    "Prophecy": ["Revelation"],
}

BOOKS = [book for group in BOOK_GROUPS.values() for book in group]
BOOK_GROUP = {book: group for group, books in BOOK_GROUPS.items() for book in books}
NEW_TESTAMENT = set(BOOKS[BOOKS.index("Matthew"):])


def testament(book):
    return "New Testament" if book in NEW_TESTAMENT else "Old Testament"
