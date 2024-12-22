import logging
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
from handlers_for_bot import start, echo


with open('secrets.txt','r') as file:
    TOKEN = file.readline().split('=')[1]



logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


# Build application
application = ApplicationBuilder().token(TOKEN).build()

# Register handlers
application.add_handler(CommandHandler('start', start))
application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), echo))

# Run the bot
application.run_polling()
