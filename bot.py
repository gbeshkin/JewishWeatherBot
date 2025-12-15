import os
import re
import math
import time
import random
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Tuple, List

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

# Требование Nominatim: валидный User-Agent (не дефолтный) + 1 rps + attribution. :contentReference[oaicite:2]{index=2}
USER_AGENT = os.getenv("USER_AGENT", "ForecastBot/1.0 (contact: you@example.com)")
NOMINATIM_BASE = os.getenv("NOMINATIM_BASE", "https://nominatim.openstreetmap.org")
GDELT_DOC_BASE = os.getenv("GDELT_DOC_BASE", "https://api.gdeltproject.org/api/v2/doc/doc")

# Релевантные слова (можно расширять)
PROTEST_QUERY = os.getenv(
    "PROTEST_QUERY",
    '"pro palestinian" OR "pro-palestinian" OR "palestine rally" OR "palestine protest" OR "pro palestine protest"'
)

# Тайм-окна (в часах) – чем ближе, тем “влажнее”
WINDOW_HOURS = [24, 72, 168]  # 1 день, 3 дня, 7 дней

# Лимит статей из GDELT
MAX_ARTICLES = int(os.getenv("MAX_ARTICLES", "50"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("protest-forecast-bot")

router = Router()


# -----------------------------
# SIMPLE CACHES (in-memory)
# -----------------------------
_geo_cache: Dict[str, Tuple[float, float, float]] = {}  # city -> (lat, lon, ts)
_geo_last_call_ts: float = 0.0


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
    return re.sub(r"\s+", " ", s.strip())


def safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return default


@dataclass
class Article:
    title: str
    url: str
    seendate: datetime
    source_country: Optional[str]


# -----------------------------
# GEO (Nominatim)
# -----------------------------
async def geocode_city(session: aiohttp.ClientSession, city: str) -> Optional[Tuple[float, float]]:
    """
    Returns (lat, lon) for a city via Nominatim.
    Respects Nominatim usage policy: 1 req/s + custom UA. :contentReference[oaicite:3]{index=3}
    """
    global _geo_last_call_ts

    key = city.lower()
    cached = _geo_cache.get(key)
    if cached and (time.time() - cached[2] < 24 * 3600):
        return (cached[0], cached[1])

    # throttle to <= 1 rps
    delta = time.time() - _geo_last_call_ts
    if delta < 1.05:
        await asyncio.sleep(1.05 - delta)

    params = {
        "q": city,
        "format": "json",
        "limit": "1",
    }
    headers = {"User-Agent": USER_AGENT}

    url = f"{NOMINATIM_BASE}/search"
    async with session.get(url, params=params, headers=headers, timeout=20) as resp:
        _geo_last_call_ts = time.time()
        resp.raise_for_status()
        data = await resp.json()

    if not data:
        return None

    lat = float(data[0]["lat"])
    lon = float(data[0]["lon"])
    _geo_cache[key] = (lat, lon, time.time())
    return (lat, lon)


# -----------------------------
# GDELT (DOC API)
# -----------------------------
def _gdelt_start_datetime(hours_back: int) -> str:
    # формат: YYYYMMDDHHMMSS
    dt = now_utc() - timedelta(hours=hours_back)
    return dt.strftime("%Y%m%d%H%M%S")


def _parse_seendate(s: str) -> datetime:
    # пример часто: "2025-12-15 12:34:56.000"
    # делаем максимально терпимо
    s = s.replace("T", " ").replace("Z", "")
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
    Pulls news articles matching query AND city name for a recent window.
    GDELT DOC API supports query + startdatetime + sort + format=json. :contentReference[oaicite:4]{index=4}
    """
    startdt = _gdelt_start_datetime(hours_back)

    # Простая логика: (protest terms) AND (city name)
    # Можно улучшать: добавлять country / language / domain filters.
    query = f"({PROTEST_QUERY}) AND ({city})"

    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "sort": "DateDesc",
        "maxrecords": str(max_articles),
        "startdatetime": startdt,
    }

    async with session.get(GDELT_DOC_BASE, params=params, headers={"User-Agent": USER_AGENT}, timeout=25) as resp:
        resp.raise_for_status()
        data = await resp.json()

    arts = []
    for item in data.get("articles", []) or []:
        title = item.get("title") or ""
        url = item.get("url") or ""
        seendate = _parse_seendate(item.get("seendate", ""))
        source_country = item.get("sourceCountry")
        if title and url:
            arts.append(Article(title=title, url=url, seendate=seendate, source_country=source_country))
    return arts


# -----------------------------
# SCORING -> "weather"
# -----------------------------
def score_from_articles(articles_by_window: Dict[int, List[Article]]) -> Dict[str, float]:
    """
    Produces numeric metrics 0..1:
    - precipitation: how many signals in near time windows
    - wind: volatility (more short-window hits relative to long-window)
    - pressure: international pressure proxy (diversity of source countries)
    """
    counts = {h: len(articles_by_window.get(h, [])) for h in WINDOW_HOURS}

    # precipitation: saturated by volume in the last 72h (signals)
    precip_raw = counts[72] + 0.5 * counts[24]
    precipitation = clamp(1.0 - math.exp(-precip_raw / 6.0), 0.0, 1.0)

    # wind: short-term spikes vs baseline
    base = max(1, counts[168])
    wind_raw = (counts[24] + 1) / (base + 1)
    wind = clamp(sigmoid((wind_raw - 1.0) * 2.2), 0.0, 1.0)

    # pressure: diversity of sources in 7d window
    countries = [a.source_country for a in articles_by_window.get(168, []) if a.source_country]
    diversity = len(set(countries))
    pressure = clamp(1.0 - math.exp(-diversity / 6.0), 0.0, 1.0)

    # temperature of opinion: a blend
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

    # “температура” 0..1 -> словарь
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

    # “погодная” метафора (без конкретных адресов — только вероятность/сигналы)
    advice = random.choice([
        "держать зонт критического мышления",
        "не читать ленту натощак",
        "проверять источники перед перепостом",
        "плотнее застёгивать куртку здравого смысла",
    ])

    # короткий блок ссылок (чтобы было прозрачно, откуда сигнал)
    links = ""
    if top_articles:
        lines = []
        for a in top_articles[:5]:
            # Telegram нормально кликает URL как текст
            lines.append(f"• {a.title}\n  {a.url}")
        links = "\n\nСигналы из открытых источников:\n" + "\n".join(lines)

    return (
        f"☁️ Прогноз общественной погоды: {city}\n\n"
        f"Утром вероятны локальные \"осадки\" из уличной повестки (вероятность: **{w['precip']}**).\n"
        f"Порывы заголовков: **{w['wind']}** — {advice}.\n\n"
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
        "Напиши команду так:\n"
        "• /forecast Tallinn\n\n"
        "Я оценю вероятность и интенсивность уличной повестки по открытым сигналам (новости/анонсы) "
        "и переведу это в \"погодные\" критерии."
    )


@router.message(Command("forecast"))
async def cmd_forecast(message: Message):
    # ожидаем: /forecast tallinn
    text = message.text or ""
    parts = text.split(maxsplit=1)
    if len(parts) < 2:
        await message.answer("Формат: /forecast <city>\nПример: /forecast Tallinn")
        return

    city = human_city(parts[1])

    async with aiohttp.ClientSession() as session:
        coords = await geocode_city(session, city)
        if not coords:
            await message.answer(f"Не смог найти город «{city}». Попробуй по-другому (например: Tallinn, Estonia).")
            return

        articles_by_window: Dict[int, List[Article]] = {}
        for h in WINDOW_HOURS:
            try:
                articles_by_window[h] = await gdelt_fetch_articles(session, city=city, hours_back=h, max_articles=MAX_ARTICLES)
            except Exception as e:
                logger.warning("GDELT fetch failed for %sh: %r", h, e)
                articles_by_window[h] = []

    metrics = score_from_articles(articles_by_window)

    # top articles: берём самые свежие за 72 часа
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
