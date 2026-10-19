"""Guards that keep small local models from acting when nobody asked (found with llama3.2 3B in live probes)."""

from __future__ import annotations

import pytest

from app.services.intent import grounded, looks_like_action, looks_like_tool_json, resolve_choice, text_tool_call
from app.tools.windows_agent import SpotifyCommand


@pytest.mark.parametrize("text", [
    "Hey NOVA, open Spotify and play Tiakola — Vie de Pirate.",
    "Can you open spotify and play tiakola vie de pirate",
    "pause the music",
    "open the calculator",
    "turn on the kitchen lights",
    "put on my coding playlist",
    "skip this song",
    "Nova, mets Vie de Pirate de Tiakola",
    "could you run the tests in this project?",
    "unlock the front door please",
    "summarize the visible page",
    "find this phrase on the page",
])
def test_requests_to_act_are_recognised(text):
    assert looks_like_action(text)


@pytest.mark.parametrize("text", [
    "What are the trade-offs between SSE and WebSockets?",
    "Explain how photosynthesis works in two sentences.",
    "How do I open a file in Python?",
    "hi",
    "Review this plan and tell me what could go wrong",
    "Pourquoi le ciel est bleu ?",
])
def test_questions_and_chat_are_not_actions(text):
    assert not looks_like_action(text)


def test_tool_calls_must_be_grounded_in_the_request():
    lights = "turn on the kitchen lights"
    assert not grounded("media_control", {"action": "volume_up"}, lights)
    assert not grounded("spotify_control", {"action": "pause"}, lights)
    assert grounded("home_control", {"entity_id": "light.kitchen", "action": "turn_on"}, lights)
    assert not grounded("home_control", {"entity_id": "lock.front_door", "action": "unlock"}, lights)
    photosynthesis = "Explain how photosynthesis works in two sentences."
    assert not grounded("open_website", {"url": "https://en.wikipedia.org/wiki/Photosynthesis"}, photosynthesis)
    assert grounded("open_website", {"url": "https://github.com/x"}, "open github")
    assert grounded("browser_inspect", {}, "summarize the visible page")
    assert grounded("browser_search", {"query": "title"}, "find this phrase on the page")
    assert grounded("open_app", {"app": "calculator"}, "open the calculator")
    assert not grounded("open_app", {"app": "notepad"}, "open the calculator")
    assert grounded("spotify_control", {"action": "play_track", "query": "Vie de Pirate"},
                    "Hey NOVA, open Spotify and play Tiakola — Vie de Pirate.")
    assert grounded("unknown_tool", {}, "anything")  # unknown tools are refused by the registry instead


def test_spotify_open_with_a_song_means_play_that_song():
    cmd = SpotifyCommand.model_validate({"action": "open", "query": "Tiakola — Vie de Pirate"})
    assert cmd.action == "play_track" and cmd.query == "Tiakola — Vie de Pirate"
    assert SpotifyCommand.model_validate({"action": "play", "query": "Starboy"}).action == "play_track"
    assert SpotifyCommand.model_validate({"action": "stop"}).action == "pause"
    assert SpotifyCommand.model_validate({"action": "open"}).action == "open"
    # Live finding: llama3.2 sent artist="null" (a string), which produced "… by null".
    cmd = SpotifyCommand.model_validate({"action": "play_track", "query": "Vie de Pirate", "artist": "null"})
    assert cmd.artist is None
    assert SpotifyCommand.model_validate({"action": "pause", "query": "None"}).query is None


CHOICES = [{"uri": "spotify:track:coeur", "name": "Cœur De Pirate", "artists": ["Tiakola"]},
           {"uri": "spotify:track:mode", "name": "Mode de vie", "artists": ["Tiakola"]},
           {"uri": "spotify:track:af1", "name": "AF1 NOUVO", "artists": ["Tiakola"]}]


@pytest.mark.parametrize("answer,index", [
    ("the first one", 0), ("1", 0), ("number two", 1), ("le premier", 0), ("the third", 2), ("the last one", 2),
    ("Mode de vie", 1), ("play coeur de pirate please", 0), ("AF1", 2),
])
def test_an_answer_picks_one_of_the_offered_choices(answer, index):
    assert resolve_choice(answer, CHOICES) == index


@pytest.mark.parametrize("answer", ["something else entirely", "Tiakola", "never mind", ""])
def test_unclear_answers_are_left_to_the_model(answer):
    assert resolve_choice(answer, CHOICES) is None


def test_text_imitating_a_tool_call_is_recognised_only_for_registered_tools():
    names = {"spotify_control"}
    call = text_tool_call('{"name": "spotify_control", "parameters": {"action": "pause"}}', names)
    assert call is not None and call.name == "spotify_control" and call.arguments == {"action": "pause"}
    assert text_tool_call('Sure! {"name": "spotify_control", "arguments": {"action": "next"}}', names).arguments == {"action": "next"}
    assert text_tool_call('{"name": "play_track", "parameters": {"track_id": "x"}}', names) is None
    assert text_tool_call("Photosynthesis uses light.", names) is None
    assert looks_like_tool_json('It seems so.\n\n {"name": "play_track", "parameters": {"track_id": "x"}}')
    assert not looks_like_tool_json("Use JSON like {\"a\": 1} in configs.")
