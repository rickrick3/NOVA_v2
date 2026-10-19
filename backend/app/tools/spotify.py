"""Spotify OAuth PKCE, DPAPI token storage, and user-authorized Web API playback.

Playback is reported as successful only after Spotify's own player state shows the requested item playing. Every
other outcome is reported precisely: not connected, no playback device, Premium required, ambiguous search, rate
limited, or accepted-but-unconfirmed.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import sys
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

import httpx

from app.tools import desktop_os
from app.tools.registry import ToolContext

AUTH = "https://accounts.spotify.com/authorize"
TOKEN = "https://accounts.spotify.com/api/token"  # noqa: S105 - OAuth endpoint, not a credential
API = "https://api.spotify.com/v1"
SCOPES = "user-read-playback-state user-modify-playback-state playlist-read-private playlist-read-collaborative"
CONNECT_URL = "/api/spotify/connect"
DEVICE_WAIT_SECONDS = 12.0  # the desktop app needs a few seconds to register as a playback device
VERIFY_SECONDS = 6.0  # Spotify's reported player state lags the command by up to a few seconds
CONTROL_ACTIONS = ("pause", "resume", "play_pause", "next", "previous", "volume_up", "volume_down")


class SpotifyProblem(Exception):
    """Safe, user-facing integration failure."""


class DpapiTokenStore:
    """Protect tokens with Windows DPAPI for the current user; never write plaintext token files."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @staticmethod
    def _crypt(data: bytes, protect: bool) -> bytes:
        if sys.platform != "win32":
            raise SpotifyProblem("Secure Spotify token storage is available only on Windows in this build.")
        import ctypes
        from ctypes import wintypes

        class Blob(ctypes.Structure):
            _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_byte))]

        raw = ctypes.create_string_buffer(data)
        source = Blob(len(data), ctypes.cast(raw, ctypes.POINTER(ctypes.c_byte)))
        output = Blob()
        crypt = ctypes.windll.crypt32.CryptProtectData if protect else ctypes.windll.crypt32.CryptUnprotectData
        ok = crypt(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(output))
        if not ok:
            raise SpotifyProblem("Windows could not protect the Spotify token for this account.")
        try:
            return ctypes.string_at(output.pbData, output.cbData)
        finally:
            ctypes.windll.kernel32.LocalFree(output.pbData)

    def save(self, tokens: dict[str, object]) -> None:
        protected = self._crypt(json.dumps(tokens, separators=(",", ":")).encode("utf-8"), True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_bytes(protected)
        os.replace(temporary, self.path)

    def load(self) -> dict[str, object] | None:
        if not self.path.is_file():
            return None
        try:
            plain = self._crypt(self.path.read_bytes(), False)
            value = json.loads(plain)
            return value if isinstance(value, dict) else None
        except (OSError, ValueError, json.JSONDecodeError, SpotifyProblem):
            return None

    def delete(self) -> None:
        self.path.unlink(missing_ok=True)


@dataclass
class PendingOAuth:
    owner_id: int
    verifier: str
    expires_at: float


class SpotifyService:
    def __init__(self, client_id: str, port: int, token_path: Path, *, transport: httpx.BaseTransport | None = None,
                 launcher: Callable[[str], bool] | None = None, processes: Callable[[], set[str]] | None = None,
                 poll_interval: float = 0.5) -> None:
        self.client_id = client_id.strip()
        self.launcher = launcher or desktop_os.launch
        self.processes = processes or desktop_os.running_processes
        self.poll_interval = poll_interval
        self.redirect_uri = f"http://127.0.0.1:{port}/api/spotify/callback"
        self.tokens = DpapiTokenStore(token_path)
        self.pending: dict[str, PendingOAuth] = {}
        self.transport = transport
        self.client = httpx.Client(transport=transport, follow_redirects=False)

    @property
    def configured(self) -> bool:
        return bool(self.client_id)

    def begin(self, owner_id: int) -> str:
        if not self.configured:
            raise SpotifyProblem("Set NOVA_SPOTIFY_CLIENT_ID and configure the callback URI in Spotify Developer settings.")
        verifier = secrets.token_urlsafe(64)[:96]
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
        state = secrets.token_urlsafe(32)
        self.pending[state] = PendingOAuth(owner_id, verifier, time.time() + 600)
        for key, item in list(self.pending.items()):
            if item.expires_at < time.time():
                self.pending.pop(key, None)
        return (
            AUTH
            + "?"
            + urlencode(
                {
                    "client_id": self.client_id,
                    "response_type": "code",
                    "redirect_uri": self.redirect_uri,
                    "scope": SCOPES,
                    "state": state,
                    "code_challenge_method": "S256",
                    "code_challenge": challenge,
                }
            )
        )

    def finish(self, code: str, state: str) -> int:
        pending = self.pending.pop(state, None)
        if pending is None or pending.expires_at < time.time():
            raise SpotifyProblem("Spotify authorization expired or its state did not match. Connect again.")
        try:
            response = self._http().post(
                TOKEN,
                data={
                    "client_id": self.client_id,
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": self.redirect_uri,
                    "code_verifier": pending.verifier,
                },
                timeout=12,
            )
        except httpx.HTTPError as exc:
            message = "Spotify authorization could not be reached. Check your internet connection and try again."
            raise SpotifyProblem(message) from exc
        if response.status_code != 200:
            raise SpotifyProblem("Spotify could not complete authorization. Check the callback URL and try again.")
        body = response.json()
        if not body.get("access_token") or not body.get("refresh_token"):
            raise SpotifyProblem("Spotify returned an incomplete authorization response.")
        self.tokens.save(
            {
                "access_token": body["access_token"],
                "refresh_token": body["refresh_token"],
                "expires_at": time.time() + int(body.get("expires_in", 3600)),
                "scope": body.get("scope", ""),
            }
        )
        return pending.owner_id

    def status(self) -> dict[str, object]:
        token = self.tokens.load() if sys.platform == "win32" else None
        return {
            "configured": self.configured,
            "connected": bool(token),
            "reason": None if self.configured else "NOVA_SPOTIFY_CLIENT_ID is not configured.",
        }

    def disconnect(self) -> None:
        self.tokens.delete()

    def close(self) -> None:
        self.client.close()

    def _http(self) -> httpx.Client:
        return self.client

    # ---- commands -----------------------------------------------------------------------------------------
    def command(self, action: str, query: str | None = None, *, artist: str | None = None, uri: str | None = None,
                ctx: ToolContext | None = None) -> tuple[str, str, dict[str, Any]]:
        """Run one Spotify operation. Raises ToolCancelled/ToolDeadline when stopped; everything else is a result."""
        ctx = ctx or ToolContext(deadline=time.monotonic() + 30)
        if not self.configured:
            return ("unavailable", "Spotify is not configured: add your Spotify app's client ID as NOVA_SPOTIFY_CLIENT_ID "
                    "in .env and restart NOVA.", {"verified": False, "blocker": "not_configured"})
        if sys.platform != "win32":
            return "unavailable", "Spotify control is only available on Windows in this build.", {"verified": False}
        try:
            if action == "open":
                return self._open(ctx)
            if action == "search":
                return self._open_search(query or "")
            token = self._access_token(ctx)
            if action in ("play_track", "play_playlist") and uri:
                return self._play_uri(token, uri, f"{query} by {artist}" if query and artist else query, ctx)
            if action == "play_track":
                return self._play_track(token, query or "", artist, ctx)
            if action == "play_playlist":
                return self._play_playlist(token, query or "", ctx)
            if action in CONTROL_ACTIONS:
                return self._control(token, action, ctx)
        except _Blocked as blocked:
            return blocked.status, blocked.message, {"verified": False, **blocked.data}
        return "unavailable", "Spotify does not support that operation.", {"verified": False}

    # ---- app ------------------------------------------------------------------------------------------------
    def _open(self, ctx: ToolContext) -> tuple[str, str, dict[str, Any]]:
        if not self.launcher("spotify:"):
            return ("unavailable", "Windows could not open Spotify. Install the Spotify desktop app and try again.",
                    {"verified": False, "blocker": "not_installed"})
        until = min(ctx.deadline - 0.2, time.monotonic() + 8)
        while True:
            if "spotify.exe" in self.processes():
                return "succeeded", "Opened Spotify.", {"verified": True}
            if time.monotonic() + 0.5 >= until:
                return ("unverified", "Asked Windows to open Spotify, but NOVA could not see the app start.",
                        {"verified": False})
            ctx.sleep(0.5)

    def _open_search(self, query: str) -> tuple[str, str, dict[str, Any]]:
        if not self.launcher(f"spotify:search:{quote(query, safe='')}"):
            return "unavailable", "Windows could not open Spotify. Install the Spotify desktop app and try again.", {}
        return ("unverified", f"Opened Spotify search for “{query}”. Choose a result there to play it.",
                {"verified": False, "operation": "search_opened"})

    # ---- playback -------------------------------------------------------------------------------------------
    def _play_track(self, token: str, query: str, artist: str | None, ctx: ToolContext) -> tuple[str, str, dict[str, Any]]:
        # Plain text, no artist: filter: models sometimes swap title and artist, and the matcher below uses every word.
        q = _clean(f"{query} {artist or ''}")
        asked = f"{query} by {artist}" if artist else query
        # No market=from_token: it needs the user-read-private scope, which NOVA does not request (Spotify answers 403).
        found = _json(self._request(token, "GET", "/search", ctx, params={"q": q, "type": "track", "limit": 10}))
        items = [t for t in found.get("tracks", {}).get("items", []) if isinstance(t, dict) and t.get("uri")]
        if not items:
            raise _Blocked("unavailable", f"Spotify found no track for “{asked}”. Try another title or artist.",
                           {"blocker": "not_found"})
        wanted = _tokens(f"{query} {artist or ''}")
        track = _best_track(items, wanted)
        if track is not None:
            chosen = track
            return self._start(token, ctx, {"uris": [chosen["uri"]]}, _track_label(chosen), chosen["uri"],
                               lambda s: _same_track(s.get("item"), chosen), {"track": _summary(chosen)})
        artist_hit = _artist_request(items, wanted)
        if artist_hit is not None:
            uri, name = artist_hit
            return self._start(token, ctx, {"context_uri": uri}, f"music by {name}", uri,
                               lambda s: (s.get("context") or {}).get("uri") == uri
                               or any(a.get("uri") == uri for a in (s.get("item") or {}).get("artists", [])),
                               {"artist": name})
        choices = items[:3]
        listing = "; ".join(f"{i}. {_track_label(t)}" for i, t in enumerate(choices, 1))
        raise _Blocked("needs_input", f"I'm not sure which track you mean by “{asked}”. Spotify's top results: {listing}. "
                       "Which one should I play?", {"candidates": [_summary(t) for t in choices]})

    def _play_uri(self, token: str, uri: str, name: str | None, ctx: ToolContext) -> tuple[str, str, dict[str, Any]]:
        """Play an exact item NOVA listed earlier (the owner chose it), without searching again."""
        label = name or "the item you chose"
        if uri.startswith("spotify:track:"):
            track = {"uri": uri, "id": uri.rsplit(":", 1)[1], "name": label}
            return self._start(token, ctx, {"uris": [uri]}, label, uri, lambda s: _same_track(s.get("item"), track),
                               {"track": {"uri": uri, "name": name}})
        return self._start(token, ctx, {"context_uri": uri}, label, uri,
                           lambda s: (s.get("context") or {}).get("uri") == uri
                           or any(a.get("uri") == uri for a in (s.get("item") or {}).get("artists", [])), {"uri": uri})

    def _play_playlist(self, token: str, query: str, ctx: ToolContext) -> tuple[str, str, dict[str, Any]]:
        found = _json(self._request(token, "GET", "/me/playlists", ctx, params={"limit": 50}))
        playlists = [p for p in found.get("items", []) if isinstance(p, dict) and p.get("uri")]
        wanted = _tokens(query)
        exact = [p for p in playlists if _tokens(p.get("name", "")) == wanted]
        matches = exact or [p for p in playlists if wanted and wanted <= _tokens(p.get("name", ""))]
        if not matches:
            raise _Blocked("unavailable", f"No playlist named “{query}” was found in your Spotify library.",
                           {"blocker": "not_found"})
        if len(matches) > 1:
            names = "; ".join(str(p.get("name", "?")) for p in matches[:5])
            raise _Blocked("needs_input", f"Several of your playlists match “{query}”: {names}. Which one should I play?",
                           {"candidates": [{"uri": p["uri"], "name": p.get("name")} for p in matches[:5]]})
        uri, name = matches[0]["uri"], matches[0].get("name", "playlist")
        return self._start(token, ctx, {"context_uri": uri}, f"your playlist {name}", uri,
                           lambda s: (s.get("context") or {}).get("uri") == uri, {"playlist": name})

    def _start(self, token: str, ctx: ToolContext, body: dict[str, Any], label: str, open_uri: str,
               playing: Callable[[dict[str, Any]], bool], extra: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
        device = self._ensure_device(token, ctx)
        for attempt in range(2):
            response = self._request(token, "PUT", "/me/player/play", ctx, params={"device_id": device["id"]},
                                     json=body, allow=(403, 404))
            if response.status_code == 404 and attempt == 0:
                ctx.sleep(1.0)  # the device may still be starting up: look it up again once
                device = self._ensure_device(token, ctx)
                continue
            if response.status_code == 403:
                if _reason(response) == "PREMIUM_REQUIRED":
                    opened = self.launcher(open_uri)
                    message = "Spotify Premium is required for NOVA to start playback. " + (
                        f"I opened {label} in the Spotify app instead; press play there." if opened else
                        "Open Spotify and start it there.")
                    return "unverified", message, {"verified": False, "blocker": "premium_required", **extra}
                raise _Blocked("failed", "Spotify refused to start playback. Check the account's plan and the permissions "
                               "granted to NOVA.", {"blocker": "forbidden"})
            if response.status_code == 404:
                raise _Blocked("unavailable", "Spotify lost the playback device. Keep the Spotify app open and try again.",
                               {"blocker": "no_device"})
            break
        state = self._verify(token, ctx, lambda s: bool(s.get("is_playing")) and playing(s))
        device_name = ((state or {}).get("device") or {}).get("name") or device.get("name") or "Spotify"
        if state is None:
            return ("unverified", f"Spotify accepted the request, but NOVA could not confirm that {label} is playing.",
                    {"verified": False, "device": device_name, **extra})
        return "succeeded", f"Playing {label} on Spotify ({device_name}).", {"verified": True, "device": device_name, **extra}

    # ---- controls -------------------------------------------------------------------------------------------
    def _control(self, token: str, action: str, ctx: ToolContext) -> tuple[str, str, dict[str, Any]]:
        current = self._player(token, ctx)
        if not current:
            raise _Blocked("unavailable", "Spotify has no active playback right now. Start something in Spotify, or ask me "
                           "to play a track, first.", {"blocker": "no_playback"})
        if action == "play_pause":
            action = "pause" if current.get("is_playing") else "resume"
        before = (current.get("item") or {}).get("uri")
        if action in ("volume_up", "volume_down"):
            volume = (current.get("device") or {}).get("volume_percent")
            if volume is None:
                raise _Blocked("unavailable", "Spotify does not report a volume for the current device.",
                               {"blocker": "no_volume"})
            target = max(0, min(100, int(volume) + (10 if action == "volume_up" else -10)))
            self._request(token, "PUT", "/me/player/volume", ctx, params={"volume_percent": target})
            if self._verify(token, ctx, lambda s: (s.get("device") or {}).get("volume_percent") == target) is None:
                return "unverified", "Spotify accepted the volume change, but NOVA could not confirm it.", {"verified": False}
            return "succeeded", f"Spotify volume set to {target}%.", {"verified": True, "volume": target}
        steps: dict[str, tuple[str, str, Callable[[dict[str, Any]], bool], str]] = {
            "pause": ("PUT", "/me/player/pause", lambda s: not s.get("is_playing"), "Paused Spotify."),
            "resume": ("PUT", "/me/player/play", lambda s: bool(s.get("is_playing")), "Resumed Spotify."),
            "next": ("POST", "/me/player/next", lambda s: (s.get("item") or {}).get("uri") not in (None, before),
                     "Skipped to the next track"),
            "previous": ("POST", "/me/player/previous",
                         lambda s: (s.get("item") or {}).get("uri") != before or int(s.get("progress_ms") or 0) < 5000,
                         "Went back to the previous track"),
        }
        method, path, check, done = steps[action]
        self._request(token, method, path, ctx)
        state = self._verify(token, ctx, check)
        if state is None:
            return "unverified", f"Spotify accepted the request ({action}), but NOVA could not confirm it.", {"verified": False}
        if action in ("next", "previous"):
            done = f"{done}: {_track_label(state['item'])}." if state.get("item") else f"{done}."
        return "succeeded", done, {"verified": True}

    # ---- devices and state ----------------------------------------------------------------------------------
    def _ensure_device(self, token: str, ctx: ToolContext) -> dict[str, Any]:
        device = _pick_device(self._devices(token, ctx))
        if device is not None:
            return device
        # The Spotify desktop app registers itself as a playback device once it is running.
        launched = self.launcher("spotify:")
        until = min(ctx.deadline - 1.0, time.monotonic() + DEVICE_WAIT_SECONDS)
        while launched and time.monotonic() + self.poll_interval * 2 < until:
            ctx.sleep(self.poll_interval * 2)
            device = _pick_device(self._devices(token, ctx))
            if device is not None:
                return device
        if not launched:
            raise _Blocked("unavailable", "There is no Spotify playback device and Windows could not open the Spotify app. "
                           "Install or open Spotify on a device, then try again.", {"blocker": "no_device"})
        raise _Blocked("unavailable", "I opened the Spotify app, but no Spotify playback device appeared. Make sure you are "
                       "signed in to Spotify on this PC (or open Spotify on another device), then ask again.",
                       {"blocker": "no_device"})

    def _devices(self, token: str, ctx: ToolContext) -> list[dict[str, Any]]:
        devices = _json(self._request(token, "GET", "/me/player/devices", ctx)).get("devices", [])
        return [d for d in devices if isinstance(d, dict)]

    def _player(self, token: str, ctx: ToolContext) -> dict[str, Any]:
        return _json(self._request(token, "GET", "/me/player", ctx, allow=(204,)))

    def _verify(self, token: str, ctx: ToolContext, predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any] | None:
        """Poll Spotify's reported player state until it matches, for a few seconds at most."""
        until = min(ctx.deadline - 0.2, time.monotonic() + VERIFY_SECONDS)
        while True:
            state = self._player(token, ctx)
            if state and predicate(state):
                return state
            if time.monotonic() + self.poll_interval >= until:
                return None
            ctx.sleep(self.poll_interval)

    # ---- HTTP -----------------------------------------------------------------------------------------------
    def _access_token(self, ctx: ToolContext) -> str:
        tokens = self.tokens.load()
        if not tokens:
            raise _Blocked("needs_authorization", "Spotify isn't connected to NOVA yet. Connect your account (Settings → Voice "
                           "→ Spotify → Connect Spotify), then ask again.",
                           {"connect_url": CONNECT_URL, "blocker": "not_connected"})
        if float(tokens.get("expires_at", 0)) < time.time() + 30:
            ctx.check()
            try:
                response = self._http().post(TOKEN, data={"client_id": self.client_id, "grant_type": "refresh_token",
                                                          "refresh_token": tokens.get("refresh_token")}, timeout=12)
            except httpx.HTTPError:
                raise _Blocked("failed", "Spotify could not be reached to refresh authorization. Check your internet "
                               "connection and try again.", {"blocker": "network"}) from None
            if response.status_code != 200:
                self.tokens.delete()
                raise _Blocked("needs_authorization", "Spotify authorization expired. Reconnect your account in Settings → "
                               "Voice → Spotify.", {"connect_url": CONNECT_URL, "blocker": "expired"})
            body = response.json()
            tokens["access_token"] = body["access_token"]
            tokens["expires_at"] = time.time() + int(body.get("expires_in", 3600))
            tokens["refresh_token"] = body.get("refresh_token", tokens["refresh_token"])
            self.tokens.save(tokens)
        return str(tokens["access_token"])

    def _request(self, token: str, method: str, path: str, ctx: ToolContext, *, allow: tuple[int, ...] = (),
                 **kwargs: Any) -> httpx.Response:
        """One API call with a bounded retry for rate limits and server errors."""
        for attempt in range(2):
            ctx.check()
            try:
                response = self._http().request(method, API + path, headers={"Authorization": f"Bearer {token}"},
                                                timeout=max(1.0, min(10.0, ctx.remaining())), **kwargs)
            except httpx.TimeoutException:
                if attempt == 0 and ctx.remaining() > 3:
                    continue
                raise _Blocked("failed", "Spotify did not answer in time. Try again in a moment.",
                               {"blocker": "timeout"}) from None
            except httpx.HTTPError:
                raise _Blocked("failed", "Spotify could not be reached. Check your internet connection and try again.",
                               {"blocker": "network"}) from None
            if response.status_code == 429:
                wait = _retry_after(response)
                if attempt == 0 and wait <= min(3.0, ctx.remaining() - 1):
                    ctx.sleep(wait)
                    continue
                raise _Blocked("failed", "Spotify is rate limiting NOVA right now. Wait a moment and try again.",
                               {"blocker": "rate_limited"})
            if response.status_code >= 500 and attempt == 0:
                ctx.sleep(0.5)
                continue
            if response.status_code in allow:
                return response
            if response.status_code == 401:
                raise _Blocked("needs_authorization", "Spotify rejected NOVA's authorization. Reconnect your account in "
                               "Settings → Voice → Spotify.", {"connect_url": CONNECT_URL, "blocker": "unauthorized"})
            if response.status_code == 403:
                raise _Blocked("failed", "Spotify denied this request. Check the account's plan and the permissions granted "
                               "to NOVA.", {"blocker": _reason(response).lower() or "forbidden"})
            if response.status_code >= 400:
                raise _Blocked("failed", f"Spotify could not complete the request (HTTP {response.status_code}).",
                               {"blocker": "api_error"})
            return response
        raise _Blocked("failed", "Spotify kept failing; try again later.", {"blocker": "api_error"})


class _Blocked(Exception):
    """A precise, user-facing reason that ends the current operation."""

    def __init__(self, status: str, message: str, data: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.data = data or {}


# ---- matching helpers -------------------------------------------------------------------------------------
def _tokens(text: str) -> frozenset[str]:
    plain = unicodedata.normalize("NFKD", text.casefold().replace("œ", "oe").replace("æ", "ae"))
    plain = "".join(c for c in plain if not unicodedata.combining(c)).replace("'", "")
    return frozenset(re.findall(r"[a-z0-9]+", plain))


def _clean(text: str) -> str:
    """A search phrase without separators that confuse Spotify's search ('Artist — Title')."""
    return " ".join(re.sub(r"[—–\-:|/]+", " ", text).split())


def _core_title(name: str) -> str:
    name = re.sub(r"\s*[(\[].*?[)\]]", "", name)  # (Remix), [Live], (feat. X)
    return name.split(" - ")[0]


def _best_track(items: list[dict[str, Any]], wanted: frozenset[str]) -> dict[str, Any] | None:
    """The first result (Spotify's ranking) among those whose title is in the request, preferring the named artist
    and the original version. None when no title matches: then the owner chooses."""

    def artist_named(t: dict[str, Any]) -> bool:
        return any(bool(_tokens(a.get("name", ""))) and _tokens(a.get("name", "")) <= wanted for a in t.get("artists", []))

    titled = [t for t in items if _tokens(_core_title(t.get("name", ""))) and _tokens(_core_title(t.get("name", ""))) <= wanted]
    if not titled:
        return None
    ranked = sorted(enumerate(titled), key=lambda p: (not artist_named(p[1]), not _tokens(p[1].get("name", "")) <= wanted, p[0]))
    return ranked[0][1]


def _artist_request(items: list[dict[str, Any]], wanted: frozenset[str]) -> tuple[str, str] | None:
    """When the request is just an artist's name ("play Tiakola"), that artist."""
    for t in items:
        for a in t.get("artists", []):
            if a.get("uri") and _tokens(a.get("name", "")) == wanted:
                return str(a["uri"]), str(a.get("name"))
    return None


def _same_track(item: dict[str, Any] | None, track: dict[str, Any]) -> bool:
    if not item:
        return False
    linked = (item.get("linked_from") or {}).get("uri")
    return item.get("uri") == track["uri"] or linked == track["uri"] or (bool(track.get("id")) and item.get("id") == track["id"])


def _track_label(track: dict[str, Any]) -> str:
    artists = ", ".join(a.get("name", "") for a in track.get("artists", []) if a.get("name"))
    return f"{track.get('name', 'Track')} by {artists}" if artists else str(track.get("name", "Track"))


def _summary(track: dict[str, Any]) -> dict[str, Any]:
    return {"uri": track["uri"], "name": track.get("name"), "artists": [a.get("name") for a in track.get("artists", [])]}


def _pick_device(devices: list[dict[str, Any]]) -> dict[str, Any] | None:
    usable = [d for d in devices if d.get("id") and not d.get("is_restricted")]
    return (next((d for d in usable if d.get("is_active")), None)
            or next((d for d in usable if d.get("type") == "Computer"), None)
            or (usable[0] if usable else None))


def _json(response: httpx.Response) -> dict[str, Any]:
    if response.status_code == 204 or not response.content:
        return {}
    try:
        value = response.json()
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _reason(response: httpx.Response) -> str:
    error = _json(response).get("error")
    return str(error.get("reason", "")) if isinstance(error, dict) else ""


def _retry_after(response: httpx.Response) -> float:
    try:
        return max(0.0, float(response.headers.get("Retry-After", "1")))
    except ValueError:
        return 1.0

