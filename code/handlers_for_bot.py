import logging
from telegram import Update
from telegram.ext import ContextTypes
from model import generate_response

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="I'm a bot, please talk to me!")



async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_message = update.message.text.strip()
        if not user_message:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="I can't process empty messages.")
            return

        logging.info(f"Received message: {user_message}")
        response = generate_response(user_message)
        logging.info(f"Model response: {response}")

        await context.bot.send_message(chat_id=update.effective_chat.id, text=response)
    except Exception as e:
        logging.error(f"Error processing message: {e}")
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, something went wrong.")