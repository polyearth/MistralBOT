import os
import json
import asyncio
import threading
import requests
import discord
import ollama

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.align import Align

console = Console()

# ================= CONFIG =================
DISCORD_TOKEN = ""  # 🔐 set in environment
DISCORD_CHANNEL_ID = "your channel id here"

WEATHER_API_KEY = "your openweathermap api key here"
# =========================================

# ========== BANNER ==========
banner = r"""
  _  __     _     _      _    ___ 
 | |/ /_ __(_)___| |_   / \  |_ _|
 | ' /| '__| / __| __| / _ \  | | 
 | . \| |  | \__ \ |_ / ___ \ | | 
 |_|\_\_|  |_|___/\__/_/   \_\___|
"""

def boot():
    console.print(Align.center(f"[bold green]{banner}[/bold green]"))
    console.print(Panel("KristAI Agent Mode", style="bold cyan"))

boot()

# ========== LOAD DOCS ==========
console.print("[yellow]Loading documents...[/yellow]")

loader = DirectoryLoader("ai_docs", glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

db = Chroma.from_documents(documents, embeddings)

console.print(Panel("AI READY (Agent Mode)", style="bold magenta"))

# ========== TOOLS ==========
def get_weather(city):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
        r = requests.get(url).json()

        if "main" in r:
            return f"{city}: {r['main']['temp']}°C, {r['weather'][0]['description']}"
        return "❌ Weather not found"

    except Exception as e:
        return f"❌ Weather API error: {e}"


def search_docs(question):
    results = db.similarity_search(question, k=4)
    return "\n".join([d.page_content[:400] for d in results])


# ========== DECISION AGENT ==========
def decide_action(user_input):
    prompt = f"""
You are an AI agent.

Choose ONE action:

1. discord(message)
2. weather(city)
3. docs(question)
4. chat(message)

Return ONLY JSON:

{{
  "action": "...",
  "value": "..."
}}

User input:
{user_input}
"""

    try:
        response = ollama.chat(
            model="mistral",
            messages=[{"role": "user", "content": prompt}]
        )

        return json.loads(response["message"]["content"])

    except:
        return {"action": "chat", "value": user_input}


# ========== DISCORD ==========
intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

discord_ready = False
discord_channel = None


@client.event
async def on_ready():
    global discord_ready, discord_channel

    console.print(f"[green][Discord][/green] Logged in as {client.user}")

    discord_channel = await client.fetch_channel(DISCORD_CHANNEL_ID)
    discord_ready = True

    console.print("[green][Discord] Ready![/green]")


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.channel.id != DISCORD_CHANNEL_ID:
        return

    # ✅ ONLY respond if bot is mentioned
    if not client.user.mentioned_in(message):
        return

    # remove mention
    user_input = message.content.replace(f"<@{client.user.id}>", "").strip()

    console.print(f"[blue][Discord User][/blue] {user_input}")

    async with message.channel.typing():

        decision = decide_action(user_input)
        action = decision.get("action", "chat")
        value = decision.get("value", user_input)

        try:
            if action == "weather":
                result = get_weather(value)
                await message.channel.send(result)

            elif action == "docs":
                context = search_docs(value)

                response = ollama.chat(
                    model="mistral",
                    messages=[{
                        "role": "user",
                        "content": f"Answer using context:\n{context}\n\nQuestion:{value}"
                    }]
                )

                await message.channel.send(response["message"]["content"])

            elif action == "discord":
                await message.channel.send(value)

            else:
                response = ollama.chat(
                    model="mistral",
                    messages=[{"role": "user", "content": value}]
                )

                await message.channel.send(response["message"]["content"])

        except Exception as e:
            await message.channel.send(f"❌ Error: {e}")


async def send_message_to_discord(msg):
    if discord_ready and discord_channel:
        await discord_channel.send(msg)


def run_discord():
    client.run(DISCORD_TOKEN)


threading.Thread(target=run_discord, daemon=True).start()


# ========== TERMINAL ==========
while True:
    user = Prompt.ask("\n[bold green]You[/bold green]")

    if user.lower() in ["exit", "quit"]:
        console.print("[red]Stopping...[/red]")
        break

    decision = decide_action(user)
    action = decision["action"]
    value = decision["value"]

    if action == "discord":
        asyncio.run_coroutine_threadsafe(
            send_message_to_discord(value),
            client.loop
        )
        console.print(f"[green]Sent to Discord:[/green] {value}")

    elif action == "weather":
        console.print(Panel(get_weather(value), title="Weather"))

    elif action == "docs":
        context = search_docs(value)

        response = ollama.chat(
            model="mistral",
            messages=[{
                "role": "user",
                "content": f"Answer using context:\n{context}\n\nQuestion:{value}"
            }]
        )

        console.print(Panel(response["message"]["content"], title="Docs"))

    else:
        response = ollama.chat(
            model="mistral",
            messages=[{"role": "user", "content": value}]
        )

        console.print(Panel(response["message"]["content"], title="KristAI"))