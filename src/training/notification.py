import asyncio
import json
import logging
from threading import Thread, Event
from typing import Any, Set, Tuple, Optional

import discord
from discord import app_commands, Intents, InteractionResponse
from dotenv import dotenv_values


class MyClient(discord.Client):
    def __init__(self, *, intents: Intents, ready_event: Optional[Event] = None, **options: Any):
        super().__init__(intents=intents, **options)

        self.ready_event = ready_event or Event()
        self.synced = False

        self.tree = app_commands.CommandTree(self)

        self.notify_user_command = self.tree.command(name='notify',
                                                     description="Be notified in channels locations specific")(
            self.notify_user)

        self.remove_user_command = self.tree.command(name='remove', description="Remove locations specific")(
            self.remove_user)

        self.users_list: Set[Tuple[int, int, int]] = set()

        self.load_users()

    def load_users(self):
        try:
            with open('users.json', 'r') as f:
                self.users_list = set(map(tuple, json.load(f)))

                logging.info("The users list has been loaded")
                logging.info(f"The users list is: \n{self.users_list}")
        except FileNotFoundError:
            logging.warning("users.json not found - starting with empty set")
            self.users_list = set()

    def save_users(self):
        with open('users.json', 'w') as f:
            json.dump(list(self.users_list), f)

            logging.info("The users list has been saved")

    def add_user_(self, user_id, guild_id, channel_id):
        info = (user_id, guild_id, channel_id)

        already_in = info in self.users_list

        if not already_in:
            self.users_list.add(info)
            self.save_users()

        return already_in

    def remove_user_(self, user_id, guild_id, channel_id):
        info = (user_id, guild_id, channel_id)

        already_in = info not in self.users_list

        if not already_in:
            self.users_list.remove(info)
            self.save_users()

        return already_in

    async def remove_user(self, message: discord.Interaction):
        response: InteractionResponse = message.response

        already_removed = self.remove_user_(message.user.id, message.guild_id, message.channel_id)

        if already_removed:
            await  response.send_message("You have already been removed", ephemeral=True)
        else:
            await  response.send_message("You have been removed from the notification list on this channel",
                                         ephemeral=True)

    async def notify_user(self, message: discord.Interaction):
        response: InteractionResponse = message.response

        already_added = self.add_user_(message.user.id, message.guild_id, message.channel_id)

        if already_added:
            await  response.send_message("You have already been added", ephemeral=True)
        else:
            await  response.send_message("You have been added to the notification list", ephemeral=True)

    async def on_ready(self):
        logging.info(f'Logged on as {self.user} (id: {self.user.id})')

        if not self.synced:
            logging.info("Syncing application commands...")
            # try:
            # await self.tree.sync()
            # logging.info("Synced application commands.")
            # except Exception:
            #     logging.exception("Failed to sync app commands.")
            self.synced = True

        self.ready_event.set()

    async def _notify_async(self, message: str):
        logging.info(f"Starting async notify task to {len(self.users_list)} users")

        for user_id, guild_id, channel_id in self.users_list:
            # first try cached channel
            channel = self.get_channel(channel_id)
            if channel is None:
                # fetch channel over HTTP if not cached
                try:
                    channel = await self.fetch_channel(channel_id)
                except discord.NotFound:
                    logging.warning(f"Channel {channel_id} not found (skipping)")
                    continue
                except Exception:
                    logging.exception(f"Error fetching channel {channel_id} (skipping)")
                    continue

            # sending may raise exceptions, handle them
            try:
                logging.warning(f"Sending message")
                await channel.send(f'<@{user_id}> {message}')
                logging.debug(f"Sent notify to user {user_id} in channel {channel_id}")
            except discord.Forbidden:
                logging.warning(f"Forbidden to send to channel {channel_id} - removing subscription entry")
                # optional: remove entry so we don't keep failing
                try:
                    self.users_list.remove((user_id, guild_id, channel_id))
                    self.save_users()
                except KeyError:
                    pass
            except Exception:
                logging.exception(f"Failed to send message to {channel_id}")

    def notify(self, message: str):
        if not hasattr(self, "loop") or self.loop is None:
            raise RuntimeError(
                "Bot loop is not available yet; start the bot and wait for ready_event before calling notify()")

        # schedule the async method to run inside the bot’s event loop
        fut = asyncio.run_coroutine_threadsafe(self._notify_async(message), self.loop)
        return fut


class Notification:
    def __init__(self, token: str, *, intents: Optional[Intents] = None):
        self.token = token
        self.ready = Event()
        self.intents = intents or Intents.default()
        self.intents.message_content = True
        self.intents.members = True

        self.client = MyClient(intents=self.intents, ready_event=self.ready)
        self.thread = Thread(target=self._run_client, daemon=True)

    def _run_client(self):
        # This will create and run the event loop inside the thread.
        # client.run blocks until the bot is closed.
        try:
            self.client.run(self.token)
        except Exception:
            logging.exception("discord.Client.run() exited with an exception")

    def start(self, timeout: Optional[float] = 30.0):
        """
        Start the client in a background thread and wait for readiness.
        `timeout` optional; if None will wait forever.
        """
        logging.info("Starting Discord client thread...")
        self.thread.start()
        started = self.ready.wait(timeout=timeout)
        if not started:
            raise TimeoutError("Bot did not signal ready within timeout")
        logging.info("Discord client is ready.")

    def notify(self, message: str, wait: bool = False, timeout: Optional[float] = None):
        """
        Synchronous notify wrapper.
        If wait=True, block until the send finishes (or timeout).
        Returns the concurrent.futures.Future if wait=False, else returns the result of future.result().
        """
        future = self.client.notify(message)
        if wait:
            return future.result(timeout=timeout)
        return future

    def stop(self, timeout: Optional[float] = 10.0):
        """
        Stop the bot cleanly.
        Submit client.close() to the bot loop and join the thread.
        """
        if hasattr(self.client, "loop") and self.client.loop is not None:
            fut = asyncio.run_coroutine_threadsafe(self.client.close(), self.client.loop)
            try:
                fut.result(timeout=timeout)
            except Exception:
                logging.exception("Error while closing the client")
        # join thread (it should exit once loop is closed)
        self.thread.join(timeout=timeout)


def init_discord_bot():
    #  Loads the secrets configuration data from the .env file
    config = dotenv_values(".env")

    token = config.get("BOT_TOKEN")

    if not token:
        raise SystemExit("BOT_TOKEN not found in .env")

    manager = Notification(token)
    manager.start(timeout=30)

    logging.info("Discord bot initialized")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    init_discord_bot()
