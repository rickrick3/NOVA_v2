"""Spotify Web API integration against MOCKED HTTP responses.

These prove NOVA's search, device, playback, verification and error handling logic. They do not prove that a live
Spotify account works; see docs/ACTIONS.md for the live test procedure.
"""

from __future__ import annotations

import json
import time
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from app.tools.registry import ToolCancelled, ToolContext
from app.tools.spotify import SpotifyService

TIAKOLA = {"uri": "spotify:track:pirate", "id": "pirate", "name": "Vie de Pirate", "artists": [{"name": "Tiakola"}]}
REMIX = {"uri": "spotify:track:remix", "id": "remix", "name": "Vie de Pirate (Remix)", "artists": [{"name": "DJ Other"}]}
OTHER = {"uri": "spotify:track:other", "id": "other", "name": "Pirates", "artists": [{"name": "Someone"}]}
OLD = {"uri": "spotify:track:old", "id": "old", "name": "Old Song", "artists": [{"name": "Earlier"}]}
PC = {"id": "pc-1", "name": "FAREL-PC", "type": "Computer", "is_active": True, "is_restricted": False}


class MemoryTokens:
    def __init__(self):
        self.value = None

    def save(self, value):
        self.value = dict(value)

    def load(self):
        return dict(self.value) if self.value else None

    def delete(self):
        self.value = None


class FakeSpotify:
    """A small scripted stand-in for the Spotify Web API."""

    def __init__(self, *, devices=None, search=None, player_states=None, play_status=204, play_body=None,
                 devices_after_launch=None):
        self.devices = [PC] if devices is None else devices
        self.devices_after_launch = devices_after_launch
        self.launched = []
        self.search_items = [TIAKOLA] if search is None else search
        self.player_states = list(player_states or [])
        self.play_status = play_status
        self.play_body = play_body
        self.calls: list[tuple[str, str]] = []
        self.played: list[dict] = []

    def launcher(self, uri: str) -> bool:
        self.launched.append(uri)
        if self.devices_after_launch is not None:
            self.devices = self.devices_after_launch
        return True

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append((request.method, path))
        if path.endswith("/me/player/devices"):
            return httpx.Response(200, json={"devices": self.devices})
        if path.endswith("/search"):
            self.search_params = dict(request.url.params)
            return httpx.Response(200, json={"tracks": {"items": self.search_items}})
        if path.endswith("/me/player/play") and request.method == "PUT":
            self.played.append({"params": dict(request.url.params), "body": json.loads(request.content or b"{}")})
            if self.play_status >= 400:
                return httpx.Response(self.play_status, json=self.play_body or {})
            return httpx.Response(self.play_status)
        if path.endswith("/me/player") and request.method == "GET":
            if not self.player_states:
                return httpx.Response(204)
            state = self.player_states.pop(0) if len(self.player_states) > 1 else self.player_states[0]
            return httpx.Response(204) if state is None else httpx.Response(200, json=state)
        if path.endswith("/me/player/pause") or path.endswith("/me/player/next"):
            return httpx.Response(204)
        return httpx.Response(404)


def connected_service(fake: FakeSpotify, tmp_path) -> SpotifyService:
    service = SpotifyService("client-id", 8000, tmp_path / "token.bin", transport=httpx.MockTransport(fake),
                             launcher=fake.launcher, poll_interval=0.01)
    service.tokens = MemoryTokens()
    service.tokens.save({"access_token": "access", "refresh_token": "refresh", "expires_at": time.time() + 3600})
    return service


def ctx(seconds: float = 10) -> ToolContext:
    return ToolContext(deadline=time.monotonic() + seconds)


@pytest.fixture(autouse=True)
def windows(monkeypatch):
    monkeypatch.setattr("app.tools.spotify.sys.platform", "win32")


def test_pkce_authorization_is_state_bound_and_uses_verifier(tmp_path):
    seen = {}

    def fake(request: httpx.Request) -> httpx.Response:
        seen.update(parse_qs(request.content.decode()))
        return httpx.Response(200, json={"access_token": "access", "refresh_token": "refresh", "expires_in": 3600,
                                         "scope": "user-modify-playback-state"})

    service = SpotifyService("client-id", 8000, tmp_path / "token.bin", transport=httpx.MockTransport(fake))
    service.tokens = MemoryTokens()
    auth_url = service.begin(7)
    params = parse_qs(urlparse(auth_url).query)
    assert params["code_challenge_method"] == ["S256"]
    assert service.finish("one-time-code", params["state"][0]) == 7
    assert seen["code_verifier"] and service.tokens.load()["refresh_token"] == "refresh"
    with pytest.raises(Exception, match="state did not match"):
        service.finish("replay", params["state"][0])
    service.close()


def test_exact_request_plays_the_intended_track_and_verifies_reported_state(tmp_path):
    # Spotify reports the previous track first; NOVA keeps checking until the requested one is really playing.
    fake = FakeSpotify(search=[REMIX, OTHER, TIAKOLA],
                       player_states=[{"is_playing": True, "item": OLD, "device": PC},
                                      {"is_playing": True, "item": TIAKOLA, "device": PC}])
    service = connected_service(fake, tmp_path)
    status, message, data = service.command("play_track", "Tiakola — Vie de Pirate", ctx=ctx())
    assert status == "succeeded", message
    assert message == "Playing Vie de Pirate by Tiakola on Spotify (FAREL-PC)."
    assert data["verified"] is True and data["track"]["uri"] == "spotify:track:pirate"
    assert fake.played == [{"params": {"device_id": "pc-1"}, "body": {"uris": ["spotify:track:pirate"]}}]
    service.close()


def test_artist_given_separately_is_used_for_the_search_and_the_match(tmp_path):
    fake = FakeSpotify(search=[REMIX, TIAKOLA], player_states=[{"is_playing": True, "item": TIAKOLA, "device": PC}])
    service = connected_service(fake, tmp_path)
    status, _, data = service.command("play_track", "Vie de Pirate", artist="Tiakola", ctx=ctx())
    assert status == "succeeded" and data["track"]["uri"] == TIAKOLA["uri"]
    service.close()


def test_title_and_artist_swapped_by_the_model_still_find_the_track(tmp_path):
    # Live finding: llama3.2 sent query="Tiakola", artist="Vie de Pirate"; an artist: filter then found nothing.
    fake = FakeSpotify(search=[REMIX, TIAKOLA], player_states=[{"is_playing": True, "item": TIAKOLA, "device": PC}])
    service = connected_service(fake, tmp_path)
    status, _, data = service.command("play_track", "Tiakola", artist="Vie de Pirate", ctx=ctx())
    assert status == "succeeded" and data["track"]["uri"] == TIAKOLA["uri"]
    assert fake.search_params["q"] == "Tiakola Vie de Pirate"
    service.close()


def test_title_only_request_follows_spotify_ranking_among_exact_titles(tmp_path):
    starboy = {"uri": "spotify:track:1", "id": "1", "name": "Starboy", "artists": [{"name": "The Weeknd"}]}
    cover = {"uri": "spotify:track:2", "id": "2", "name": "Starboy", "artists": [{"name": "Cover Band"}]}
    fake = FakeSpotify(search=[starboy, cover], player_states=[{"is_playing": True, "item": starboy, "device": PC}])
    service = connected_service(fake, tmp_path)
    status, message, _ = service.command("play_track", "Starboy", ctx=ctx())
    assert status == "succeeded" and "The Weeknd" in message
    service.close()


def test_ambiguous_results_ask_which_track_instead_of_guessing(tmp_path):
    fake = FakeSpotify(search=[REMIX, OTHER, OLD])
    service = connected_service(fake, tmp_path)
    status, message, data = service.command("play_track", "that pirate song", ctx=ctx())
    assert status == "needs_input"
    assert [c["uri"] for c in data["candidates"]] == [REMIX["uri"], OTHER["uri"], OLD["uri"]]
    assert "Which one" in message
    assert fake.played == []
    service.close()


def test_search_needs_no_scope_beyond_what_nova_requests(tmp_path):
    # Live finding: market=from_token requires user-read-private, which NOVA does not request -> HTTP 403.
    fake = FakeSpotify(player_states=[{"is_playing": True, "item": TIAKOLA, "device": PC}])
    service = connected_service(fake, tmp_path)
    service.command("play_track", "Tiakola Vie de Pirate", ctx=ctx())
    assert "market" not in fake.search_params and fake.search_params["type"] == "track"
    service.close()


def test_misremembered_title_offers_the_real_tracks_instead_of_guessing(tmp_path):
    # Live finding: Spotify has no Tiakola track named "Vie de Pirate"; these were its real top results.
    coeur = {"uri": "spotify:track:coeur", "id": "coeur", "name": "Cœur De Pirate", "artists": [{"name": "Tiakola"}]}
    mode = {"uri": "spotify:track:mode", "id": "mode", "name": "Mode de vie", "artists": [{"name": "Tiakola"}]}
    damso = {"uri": "spotify:track:damso", "id": "damso", "name": "Coeur De Pirate", "artists": [{"name": "Damso"}]}
    fake = FakeSpotify(search=[coeur, mode, damso])
    service = connected_service(fake, tmp_path)
    status, message, data = service.command("play_track", "Tiakola — Vie de Pirate", ctx=ctx())
    assert status == "needs_input" and fake.played == []
    assert data["candidates"][0] == {"uri": "spotify:track:coeur", "name": "Cœur De Pirate", "artists": ["Tiakola"]}
    assert "1. Cœur De Pirate by Tiakola" in message
    # The owner's answer, as the model passes it on, plays the real track.
    fake.player_states = [{"is_playing": True, "item": coeur, "device": PC}]
    status, message, _ = service.command("play_track", "Cœur De Pirate", artist="Tiakola", ctx=ctx())
    assert status == "succeeded" and message == "Playing Cœur De Pirate by Tiakola on Spotify (FAREL-PC)."
    service.close()


def test_missing_authorization_asks_to_connect_with_the_supported_flow(tmp_path):
    fake = FakeSpotify()
    service = SpotifyService("client-id", 8000, tmp_path / "token.bin", transport=httpx.MockTransport(fake),
                             launcher=fake.launcher)
    service.tokens = MemoryTokens()
    status, message, data = service.command("play_track", "Vie de Pirate", ctx=ctx())
    assert status == "needs_authorization" and "Connect" in message
    assert data["connect_url"] == "/api/spotify/connect"
    assert fake.calls == []
    service.close()


def test_missing_client_id_is_reported_as_configuration_problem(tmp_path):
    service = SpotifyService("", 8000, tmp_path / "missing.bin")
    status, message, _ = service.command("play_track", "Song", ctx=ctx())
    assert status == "unavailable" and "NOVA_SPOTIFY_CLIENT_ID" in message
    service.close()


def test_rejected_authorization_asks_to_reconnect(tmp_path):
    service = SpotifyService("client-id", 8000, tmp_path / "t.bin",
                             transport=httpx.MockTransport(lambda _: httpx.Response(401)), launcher=lambda _: True)
    service.tokens = MemoryTokens()
    service.tokens.save({"access_token": "expired", "refresh_token": "refresh", "expires_at": time.time() + 3600})
    status, message, data = service.command("play_track", "Song", ctx=ctx())
    assert status == "needs_authorization" and "reconnect" in message.lower()
    assert data["connect_url"] == "/api/spotify/connect"
    service.close()


def test_no_device_launches_the_spotify_app_and_waits_for_it(tmp_path):
    fake = FakeSpotify(devices=[], devices_after_launch=[{**PC, "is_active": False}],
                       player_states=[{"is_playing": True, "item": TIAKOLA, "device": PC}])
    service = connected_service(fake, tmp_path)
    status, message, _ = service.command("play_track", "Tiakola Vie de Pirate", ctx=ctx())
    assert fake.launched == ["spotify:"]
    assert status == "succeeded", message
    service.close()


def test_no_device_even_after_launch_names_the_precise_blocker(tmp_path):
    fake = FakeSpotify(devices=[])
    service = connected_service(fake, tmp_path)
    status, message, data = service.command("play_track", "Tiakola Vie de Pirate", ctx=ctx(3))
    assert status == "unavailable" and "no Spotify playback device" in message
    assert data["blocker"] == "no_device" and fake.launched == ["spotify:"]
    assert fake.played == []
    service.close()


def test_premium_required_is_explained_and_the_track_is_opened_in_the_app(tmp_path):
    fake = FakeSpotify(play_status=403, play_body={"error": {"status": 403, "message": "Player command failed: Premium required",
                                                             "reason": "PREMIUM_REQUIRED"}})
    service = connected_service(fake, tmp_path)
    status, message, data = service.command("play_track", "Tiakola Vie de Pirate", ctx=ctx())
    assert status == "unverified" and "Premium" in message
    assert data["blocker"] == "premium_required" and data["verified"] is False
    assert "spotify:track:pirate" in fake.launched
    service.close()


def test_unconfirmed_playback_is_never_reported_as_success(tmp_path):
    fake = FakeSpotify(player_states=[{"is_playing": True, "item": OLD, "device": PC}])
    service = connected_service(fake, tmp_path)
    status, message, data = service.command("play_track", "Tiakola Vie de Pirate", ctx=ctx(2))
    assert status == "unverified" and "could not confirm" in message
    assert data["verified"] is False
    service.close()


def test_rate_limit_is_retried_once_then_reported(tmp_path):
    attempts = {"search": 0}

    def limited(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/search"):
            attempts["search"] += 1
            return httpx.Response(429, headers={"Retry-After": "0"})
        return FakeSpotify()(request)

    service = SpotifyService("client-id", 8000, tmp_path / "t.bin", transport=httpx.MockTransport(limited),
                             launcher=lambda _: True, poll_interval=0.01)
    service.tokens = MemoryTokens()
    service.tokens.save({"access_token": "a", "refresh_token": "r", "expires_at": time.time() + 3600})
    status, message, _ = service.command("play_track", "Tiakola Vie de Pirate", ctx=ctx())
    assert status == "failed" and "rate limit" in message.lower()
    assert attempts["search"] == 2
    service.close()


def test_controls_handle_no_active_playback_without_crashing(tmp_path):
    fake = FakeSpotify(player_states=[None])
    service = connected_service(fake, tmp_path)
    for action in ("pause", "play_pause", "volume_up"):
        status, message, _ = service.command(action, ctx=ctx())
        assert status == "unavailable" and "no active playback" in message.lower()
    service.close()


def test_pause_is_verified_from_reported_state(tmp_path):
    fake = FakeSpotify(player_states=[{"is_playing": True, "item": TIAKOLA, "device": PC},
                                      {"is_playing": False, "item": TIAKOLA, "device": PC}])
    service = connected_service(fake, tmp_path)
    status, message, data = service.command("pause", ctx=ctx())
    assert status == "succeeded" and message == "Paused Spotify." and data["verified"] is True
    service.close()


def test_cancellation_stops_waiting_for_a_device(tmp_path):
    fake = FakeSpotify(devices=[])
    service = connected_service(fake, tmp_path)
    c = ctx(30)
    c.cancel()
    with pytest.raises(ToolCancelled):
        service.command("play_track", "Tiakola Vie de Pirate", ctx=c)
    service.close()


def test_a_chosen_track_is_played_by_its_exact_spotify_id(tmp_path):
    fake = FakeSpotify(player_states=[{"is_playing": True, "item": TIAKOLA, "device": PC}])
    service = connected_service(fake, tmp_path)
    status, message, data = service.command("play_track", "Vie de Pirate", artist="Tiakola", uri="spotify:track:pirate",
                                            ctx=ctx())
    assert status == "succeeded" and data["verified"] is True
    assert message == "Playing Vie de Pirate by Tiakola on Spotify (FAREL-PC)."
    assert ("GET", "/v1/search") not in fake.calls
    assert fake.played[0]["body"] == {"uris": ["spotify:track:pirate"]}
    service.close()
