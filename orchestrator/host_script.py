"""
host_script.py — Host Announcer Text Generator
===============================================
DEBATE Project · Bauhaus-Universität Weimar · Webis Lab · SS 2026

Generates the scripted lines the host reads aloud via TTS at key moments:
intro, round transitions, and the final winner announcement.
"""

from __future__ import annotations


def intro(topic: str, llm_a: str, llm_b: str, turn_count: int) -> str:
    rounds = turn_count // 2
    return (
        f"Welcome to DebateArena. "
        f"Today's topic: {topic}. "
        f"In the left corner, arguing in favour: {llm_a}. "
        f"In the right corner, arguing against: {llm_b}. "
        f"The debate will run for {rounds} rounds. "
        f"Debaters, take your positions. Round one begins now."
    )


def round_transition(round_number: int) -> str:
    return f"Round {round_number}. Both speakers, please advance your argument."


def rebuttal_prompt(speaker: str) -> str:
    return f"{speaker}, your rebuttal."


def winner_announcement(winner: str, explanation: str) -> str:
    return (
        f"The debate has concluded. "
        f"After evaluating coverage, argument quality, and composure, "
        f"the winner is: {winner}. "
        f"{explanation}"
    )


def tie_announcement() -> str:
    return (
        "The debate has concluded. "
        "After careful evaluation, both speakers performed equally. "
        "This debate is declared a tie."
    )
