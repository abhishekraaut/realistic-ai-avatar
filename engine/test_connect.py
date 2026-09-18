import asyncio
import os
from livekit import api, rtc

async def main():
    room = rtc.Room()
    token = api.AccessToken('devkey', 'secret').with_identity('user').with_name('Test').with_grants(api.VideoGrants(room_join=True, room='avatar-room-test')).to_jwt()
    await room.connect('ws://localhost:7880', token)
    print("Connected to room!")
    await asyncio.sleep(5)
    await room.disconnect()

asyncio.run(main())
