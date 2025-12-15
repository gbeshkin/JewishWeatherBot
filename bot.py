import os
import re
import math
import random
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List

import aiohttp
from aiogram import Bot, Dispatcher, Router
from aiogram.filters import Command
from aiogram.types import Message

# -----------------------------
# CONFIG
# -----------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN env var is missing")

USER_AGENT = os.getenv("USER_AGENT", "JewishWeatherBot/1.0 (contact: you@example.com)")
GDELT_DOC_BASE = os.getenv("GDELT_DOC_BASE", "https://api.gdeltproject.org/api/v2/doc/doc")

# Запрос: протесты + пропалестинские. Можно расширять.
PROTEST_QUERY = os.getenv(
    "PROTEST_QUERY",
    '"pro palestinian" OR "pro-palestinian" OR "pro palestine" OR "palestine rally" OR "palestine protest" OR "pro-palestine protest"'
)

WINDOW_HOURS = [24, 72, 168]  # 1, 3, 7 дней
MAX_ARTICLES = int(os.getenv("MAX_ARTICLES", "50"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("protest-forecast-bot")

router = Router()

# -----------------------------
# MODELS
# -----------------------------
@dataclass
class Article:
    title: str
    url: str
    seendate: datetime
    source_country: Optional[str]


# -----------------------------
# HELPERS
# -----------------------------
def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def human_city(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip())
    # защита от слишком длинных промптов/мусора
    return s[:80]


def _gdelt_start_datetime(hours_back: int) -> str:
    dt = now_utc() - timedelta(hours=hours_back)
    return dt.strftime("%Y%m%d%H%M%S")


def _parse_seendate(s: str) -> datetime:
    s = (s or "").replace("T", " ").replace("Z", "")
    s = re.sub(r"\.\d+$", "", s).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y%m%d%H%M%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            pass
    return now_utc()


async def gdelt_fetch_articles(
    session: aiohttp.ClientSession,
    city: str,
    hours_back: int,
    max_articles: int = 50,
) -> List[Article]:
    """
    Ищем новости/анонсы: (пропалестинские протесты) AND (город).
    """
    startdt = _gdelt_start_datetime(hours_back)

    # Для “Tallinn, Estonia” лучше, чем просто “Tallinn”
    query = f"({PROTEST_QUERY}) AND ({city})"

    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "sort": "DateDesc",
        "maxrecords": str(max_articles),
        "startdatetime": startdt,
    }

    async with session.get(
        GDELT_DOC_BASE,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=25
    ) as resp:
        resp.raise_for_status()
        data = await resp.json()

    arts: List[Article] = []
    for item in (data.get("articles") or []):
        title = item.get("title") or ""
        url = item.get("url") or ""
        seendate = _parse_seendate(item.get("seendate", ""))
        source_country = item.get("sourceCountry")
        if title and url:
            arts.append(Article(title=title, url=url, seendate=seendate, source_country=source_country))
    return arts


def score_from_articles(articles_by_window: Dict[int, List[Article]]) -> Dict[str, float]:
    counts = {h: len(articles_by_window.get(h, [])) for h in WINDOW_HOURS}

    # “осадки”: насыщение по сигналам (72ч + вес 24ч)
    precip_raw = counts[72] + 0.5 * counts[24]
    precipitation = clamp(1.0 - math.exp(-precip_raw / 6.0), 0.0, 1.0)

    # “ветер”: краткосрочные всплески относительно 7 дней
    base = max(1, counts[168])
    wind_raw = (counts[24] + 1) / (base + 1)
    wind = clamp(sigmoid((wind_raw - 1.0) * 2.2), 0.0, 1.0)

    # “давление”: разнообразие стран-источников за 7 дней
    countries = [a.source_country for a in articles_by_window.get(168, []) if a.source_country]
    diversity = len(set(countries))
    pressure = clamp(1.0 - math.exp(-diversity / 6.0), 0.0, 1.0)

    temperature = clamp(0.55 * precipitation + 0.45 * wind, 0.0, 1.0)

    return {
        "precipitation": precipitation,
        "wind": wind,
        "pressure": pressure,
        "temperature": temperature,
        "count_24h": float(counts[24]),
        "count_72h": float(counts[72]),
        "count_7d": float(counts[168]),
    }


def weather_words(m: Dict[str, float]) -> Dict[str, str]:
    def lvl(x: float, a: float, b: float) -> str:
        if x < a:
            return "низкая"
        if x < b:
            return "умеренная"
        return "высокая"

    precip = lvl(m["precipitation"], 0.25, 0.6)
    wind = lvl(m["wind"], 0.25, 0.6)
    pressure = lvl(m["pressure"], 0.25, 0.6)

    t = m["temperature"]
    if t < 0.25:
        temp = "прохладная"
    elif t < 0.5:
        temp = "тёплая"
    elif t < 0.75:
        temp = "горячая"
    else:
        temp = "перегретая"

    return {"precip": precip, "wind": wind, "pressure": pressure, "temp": temp}


def format_forecast(city: str, metrics: Dict[str, float], top_articles: List[Article]) -> str:
    w = weather_words(metrics)

    advice = random.choice([
        "держать зонт критического мышления",
        "не читать ленту натощак",
        "проверять источники перед репостом",
        "плотнее застёгивать куртку здравого смысла",
    ])

    links = ""
    if top_articles:
        lines = []
        for a in top_articles[:5]:
            lines.append(f"• {a.title}\n  {a.url}")
        links = "\n\nСигналы из открытых источников:\n" + "\n".join(lines)

    return (
        f"☁️ Прогноз общественной погоды: {city}\n\n"
        f"Вероятность локальных \"осадков\" (анонсы/упоминания выступлений): **{w['precip']}**.\n"
        f"Порывы заголовков: **{w['wind']}** — рекомендуется {advice}.\n\n"
        f"🌡 Температура общественного мнения — **{w['temp']}**\n"
        f"🌍 Международное давление — **{w['pressure']}**\n\n"
        f"📊 Сигналы в новостях: 24ч={int(metrics['count_24h'])}, 72ч={int(metrics['count_72h'])}, 7д={int(metrics['count_7d'])}\n\n"
        f"Береги себя: даже шумная погода не отменяет свет."
        f"{links}"
    )


# -----------------------------
# ROUTES
# -----------------------------
@router.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer(
        "Формат команды:\n"
        "• /forecast Tallinn\n"
        "• /forecast Tallinn, Estonia\n\n"
        "Я ищу публичные сигналы в новостях/анонсах и перевожу их в «погодные» метрики."
    )


@router.message(Command("forecast"))
async def cmd_forecast(message: Message):
    text = message.text or ""
    parts = text.split(maxsplit=1)
    if len(parts) < 2:
        await message.answer("Формат: /forecast <city>\nПример: /forecast Tallinn")
        return

    city = human_city(parts[1])

    async with aiohttp.ClientSession() as session:
        articles_by_window: Dict[int, List[Article]] = {}
        for h in WINDOW_HOURS:
            try:
                articles_by_window[h] = await gdelt_fetch_articles(
                    session, city=city, hours_back=h, max_articles=MAX_ARTICLES
                )
            except Exception as e:
                logger.warning("GDELT fetch failed for %sh: %r", h, e)
                articles_by_window[h] = []

    metrics = score_from_articles(articles_by_window)
    top_articles = sorted(articles_by_window.get(72, []), key=lambda a: a.seendate, reverse=True)

    await message.answer(format_forecast(city, metrics, top_articles), disable_web_page_preview=True)


# -----------------------------
# MAIN
# -----------------------------
async def main():
    bot = Bot(BOT_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)
    logger.info("Bot started (polling)")
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopped")
