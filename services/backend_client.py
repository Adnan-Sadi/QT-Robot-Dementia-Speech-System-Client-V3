import json
import asyncio
import threading
from urllib.parse import urlencode
from typing import Optional
import contextlib

import aiohttp

from config.settings import settings

class BackendClient:
    """
    - Auth: POST {BASE}/api/token/ -> {'access','refresh'}
    - WS :  wss://{HOST}/ws/chat/?token=<access>&source=<client>
    - Send: {"type":"transcription","data":"..."}
    - Receive: 'llm_response', 'emotion', etc.
    """

    def __init__(self, base_http: str = None, ws_path: str = None, source: str = None):
        self.base_http = (base_http or settings.BASE_HTTP_URL).rstrip("/")
        self.ws_path = ws_path or settings.WS_PATH
        self.ws_path = self.ws_path if self.ws_path.startswith("/") else "/" + self.ws_path
        self.source = source or settings.SOURCE

        self._http: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._listen_task: Optional[asyncio.Task] = None

        self.access: Optional[str] = None
        self.refresh: Optional[str] = None
        self.ws_url: Optional[str] = None

        self._on_llm_response = None  # Set externally via BackendBridge.set_response_callback()

    # ---------------------------
    # Lifecycle
    # ---------------------------
    async def start(self):
        self._http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) # maximum number of seconds for the whole operation
        await self._login() # get token and prepare ws_url
        await self._connect_ws() # connect to websocket
        self._listen_task = asyncio.create_task(self._listen_loop()) # start listen loop

    async def stop(self):
        if self._listen_task:
            self._listen_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._listen_task
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._http:
            await self._http.close()

    async def _login(self):
        assert self._http
        url = f"{self.base_http}/api/token/"
        # Expect USERNAME/PASSWORD in environment
        username = settings.USERNAME
        password = settings.PASSWORD

        if not username or not password:
            raise RuntimeError("USERNAME and PASSWORD must be set in environment")
        # get token
        async with self._http.post(url, json={"username": username, "password": password}) as r:
            if r.status != 200:
                raise RuntimeError(f"Token request failed: {r.status} {await r.text()}")
            data = await r.json()
        # fetch the access and refresh token from the response
        # In our case, both access and refresh are the same JWT token
        self.access = data.get("access")
        self.refresh = data.get("refresh")

        if not self.access:
            raise RuntimeError(f"Missing 'access' in token response: {data}")
        # prepare the websocket url
        scheme = "wss" if self.base_http.startswith("https") else "ws"
        
        params = {"token": self.access, "source": self.source}
            
        qs = urlencode(params)
        self.ws_url = f"{scheme}://{self.base_http.split('://',1)[1]}{self.ws_path}?{qs}"

    async def _connect_ws(self):
        assert self._http and self.ws_url
        headers = {"Origin": self.base_http}
        # aiohttp uses its own ping/pong mechanism to keep the connection alive
        # if the server does not respond within 20 seconds, the connection is closed
        self._ws = await self._http.ws_connect(self.ws_url, headers=headers, heartbeat=20)

    # ---------------------------
    # Listen & dispatch
    # ---------------------------
    async def _listen_loop(self):
        assert self._ws
        while True:
            msg = await self._ws.receive()
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
            
                mtype = data.get("type")
                if mtype == "llm_response":
                    async with self._lock:
                        payload = data.get("data")
                        emotion = data.get("emotion")
                        
                        # Handle both string (ChatConsumer) and dict (ActivityChatConsumer)
                        if isinstance(payload, str):
                            text = payload
                            current_scenario = None
                            next_scenario = None
                        elif isinstance(payload, dict):
                            text = payload.get("text", "")
                            current_scenario = payload.get("current_scenario")
                            next_scenario = payload.get("next_scenario")
                        else:
                            # Skip if payload is neither string nor dict
                            continue
                        
                        if self._on_llm_response:
                            self._on_llm_response(text, emotion, current_scenario, next_scenario)

            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE):
                await self._reconnect_with_backoff()
            elif msg.type == aiohttp.WSMsgType.ERROR:
                await self._reconnect_with_backoff()

    async def _reconnect_with_backoff(self):
        # simple exponential backoff up to 30s
        delay = 1
        while True:
            try:
                await asyncio.sleep(delay)
                await self._connect_ws()
                return
            except Exception:
                delay = min(delay * 2, 30)

    async def send_audio_chunk(self, pcm_bytes: bytes, sample_rate: int = 16000):
        """Send a raw PCM audio chunk to the backend."""
        import base64
        payload = {
            "type": "audio_data",
            "data": base64.b64encode(pcm_bytes).decode("utf-8"),
            "sampleRate": sample_rate,
        }
        assert self._ws
        await self._ws.send_str(json.dumps(payload))

    async def send_trigger(self):
        """Tell the backend to flush its staged utterances and generate a response."""
        assert self._ws
        await self._ws.send_str(json.dumps({"type": "send_staged"}))


class BackendBridge:
    """
    Thread-safe facade for ROS code.
    Spins an asyncio loop in a background thread and exposes blocking methods.
    """

    def __init__(self):
        # currently hardcoded, should move to an env or config file
        base = settings.BASE_HTTP_URL
        ws_path = settings.WS_PATH
        source = settings.SOURCE
        if not base:
            raise RuntimeError("BASE must be set in .env or environment")
        self._client = BackendClient(base, ws_path, source)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._started = threading.Event()
        self._stopping = threading.Event()

    def start(self):
        self._thread.start()
        fut = asyncio.run_coroutine_threadsafe(self._client.start(), self._loop)
        fut.result()  # raise if fails
        self._started.set()

    def stop(self):
        if not self._started.is_set() or self._stopping.is_set():
            return
        self._stopping.set()
        fut = asyncio.run_coroutine_threadsafe(self._client.stop(), self._loop)
        try:
            fut.result(timeout=5)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)

    def set_response_callback(self, callback):
        """
        Register a callable invoked when the backend sends an llm_response.
        Signature: callback(text: str, emotion: str, current_scenario: str, next_scenario: str)
        """
        self._client._on_llm_response = callback

    def send_audio_chunk(self, pcm_bytes: bytes, sample_rate: int = 16000):
        """Fire-and-forget: send a raw PCM chunk to the backend."""
        if not self._started.is_set():
            return  # Silently drop if not yet connected
        asyncio.run_coroutine_threadsafe(
            self._client.send_audio_chunk(pcm_bytes, sample_rate),
            self._loop,
        )
        # Intentionally no .result() — fire-and-forget

    def send_staged(self):
        """Block briefly until the trigger message is sent."""
        if not self._started.is_set():
            raise RuntimeError("BackendBridge not started. Call start() first.")
        fut = asyncio.run_coroutine_threadsafe(self._client.send_trigger(), self._loop)
        fut.result()