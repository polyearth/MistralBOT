import pandas as pd
import ollama

FILE = r"C:\Users\Kristapsv\Desktop\Book1.xlsx"

# =========================
# LOAD + PREP DATA
# =========================
def load_data():
    df = pd.read_excel(FILE)

    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["due_date"] = pd.to_datetime(df["due_date"])
    df["paid_date"] = pd.to_datetime(df["paid_date"])

    df["delay_days"] = (df["paid_date"] - df["due_date"]).dt.days
    df["is_late"] = df["delay_days"] > 0

    return df


df = load_data()

# =========================
# BUILD CLIENT STATS
# =========================
client_stats = df.groupby("client_name").agg(
    avg_delay=("delay_days", "mean"),
    max_delay=("delay_days", "max"),
    late_rate=("is_late", "mean"),
    total=("client_id", "count"),
    total_amount=("amount", "sum")
)

client_stats["risk_score"] = client_stats["late_rate"]

# =========================
# FINANCE BRAIN FUNCTION
# =========================
def finance_context():
    return client_stats.to_string()


def ask_ai(question):
    context = finance_context()

    prompt = f"""
You are a financial AI assistant.

You ONLY use this data:

{context}

Answer the user clearly and like a financial analyst.

User question:
{question}
"""

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


# =========================
# CHAT LOOP
# =========================
print("💰 Finance AI ready (type 'exit' to stop)")

while True:
    q = input("\nYou: ")

    if q.lower() == "exit":
        break

    answer = ask_ai(q)
    print("\nAI:", answer)