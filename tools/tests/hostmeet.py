"""Helper for the phone tests: a host meeting running over /ws/record (Record-only) with joining open; yields the join token."""
import asyncio, json, ssl, httpx, websockets
B = "https://127.0.0.1:8445"
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
def login(user="boss", pw="adminpassword123"):
    c = httpx.Client(base_url=B, verify=False, timeout=120, headers={"Origin": B})
    assert c.post("/api/auth/login", json={"username": user, "password": pw}).status_code == 200
    return c, "; ".join(f"{k}={v}" for k, v in c.cookies.items()), c.cookies.get("zecret_csrf")
class Host:
    def __init__(self, title="phones"):
        self.title = title; self.msgs = []; self.mid = None
    async def __aenter__(self):
        self.c, cookie, self.csrf = login()
        self.c.put("/api/settings", headers={"X-CSRF-Token": self.csrf}, json={"allow_join": True, "guest_mic": True})
        self.ws = await websockets.connect(f"wss://127.0.0.1:8445/ws/record?mode=after&title={self.title}", ssl=ctx, additional_headers={"Cookie": cookie, "Origin": B})
        self.reader = asyncio.create_task(self._read()); self.feeder = asyncio.create_task(self._feed())
        for _ in range(100):
            if self.mid: break
            await asyncio.sleep(0.1)
        r = self.c.post(f"/api/meetings/{self.mid}/join", headers={"X-CSRF-Token": self.csrf}, json={"open": True}); assert r.status_code == 200, r.text
        urls = r.json().get("urls") or []; assert urls, r.json(); self.token = urls[0].rstrip("/").split("/")[-1]  # the room link (same QR) -> this meeting
        return self
    async def __aexit__(self, *a):
        self.feeder.cancel(); self.reader.cancel()
        try: await self.ws.close()
        except Exception: pass
    async def _read(self):
        try:
            async for m in self.ws:
                m = json.loads(m); self.msgs.append(m)
                if m["type"] == "started": self.mid = m["id"]
        except Exception: pass
    async def _feed(self):  # the room microphone keeps sending quiet noise so the meeting stays alive
        import numpy as np
        rng = np.random.default_rng(0)
        try:
            while True:
                await self.ws.send((rng.standard_normal(320) * 300).astype(np.int16).tobytes()); await asyncio.sleep(0.02)
        except Exception: pass
    async def stop(self):
        self.feeder.cancel()
        try: await self.ws.send(json.dumps({"type": "stop"})); await asyncio.wait_for(self.reader, 30)
        except Exception: pass
