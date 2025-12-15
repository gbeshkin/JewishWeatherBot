import os
import re
import math
import random
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, List

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

# Запрос можно подстроить через переменную окружения.
PROTEST_QUERY = os.getenv(
    "PROTEST_QUERY",
    '"pro palestinian" OR "pro-palestinian" OR "pro palestine" OR '
    '"palestine rally" OR "palestine protest" OR "pro-palestine protest" OR '
    '"palestinian solidarity" OR "solidarity with palestine" OR '
    '"ceasefire protest" OR "gaza protest" OR "free palestine rally"'
)

WINDOW_HOURS = 24
MAX_ARTICLES = int(os.getenv("MAX_ARTICLES", "60"))

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
    return s[:80]


def maybe(p: float) -> bool:
    return random.random() < p


def pick(items: List[str]) -> str:
    return random.choice(items)


def pickn(items: List[str], n: int) -> List[str]:
    if n <= 0:
        return []
    if n >= len(items):
        items = items[:]
        random.shuffle(items)
        return items
    return random.sample(items, n)


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


async def _gdelt_get_json_tolerant(resp: aiohttp.ClientResponse) -> dict:
    # GDELT иногда отдаёт text/html при status=200 — не падаем.
    try:
        return await resp.json(content_type=None)
    except Exception:
        try:
            body = await resp.text()
        except Exception:
            body = "<unreadable>"
        logger.warning(
            "GDELT non-JSON (status=%s ct=%s): %s",
            resp.status,
            resp.headers.get("Content-Type"),
            body[:200].replace("\n", " "),
        )
        return {}


async def gdelt_fetch_articles(session: aiohttp.ClientSession, city: str) -> List[Article]:
    startdt = _gdelt_start_datetime(WINDOW_HOURS)
    query = f"({PROTEST_QUERY}) AND ({city})"

    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "sort": "DateDesc",
        "maxrecords": str(MAX_ARTICLES),
        "startdatetime": startdt,
    }

    async with session.get(
        GDELT_DOC_BASE,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=25
    ) as resp:
        if resp.status != 200:
            txt = await resp.text()
            logger.warning("GDELT HTTP %s: %s", resp.status, txt[:200].replace("\n", " "))
            return []
        data = await _gdelt_get_json_tolerant(resp)

    arts: List[Article] = []
    for item in (data.get("articles") or []):
        title = item.get("title") or ""
        url = item.get("url") or ""
        seendate = _parse_seendate(item.get("seendate", ""))
        source_country = item.get("sourceCountry")
        if title and url:
            arts.append(Article(title=title, url=url, seendate=seendate, source_country=source_country))
    return arts


# -----------------------------
# METRICS (24h only)
# -----------------------------
def compute_metrics(articles_24h: List[Article]) -> dict:
    """
    Всё только по 24 часам: больше упоминаний -> больше «осадки/ветер/температура».
    """
    n = len(articles_24h)

    # Осадки (насыщение)
    precipitation = clamp(1.0 - math.exp(-n / 6.0), 0.0, 1.0)

    # Ветер (резкость) — чуть усилим на маленьких n, чтобы текст не был “плоский”
    wind = clamp(sigmoid((n - 4.0) * 0.7), 0.0, 1.0)

    # Давление (международность) — разнообразие источников за 24ч
    countries = [a.source_country for a in articles_24h if a.source_country]
    diversity = len(set(countries))
    pressure = clamp(1.0 - math.exp(-diversity / 4.0), 0.0, 1.0)

    # Температура (общая)
    temperature = clamp(0.55 * precipitation + 0.30 * wind + 0.15 * pressure, 0.0, 1.0)

    # Уверенность: больше сигналов -> выше
    confidence = clamp(1.0 - math.exp(-n / 5.5), 0.0, 1.0)

    return {
        "n": float(n),
        "precipitation": precipitation,
        "wind": wind,
        "pressure": pressure,
        "temperature": temperature,
        "confidence": confidence,
    }


def lvl3(x: float, a: float, b: float, low: str, mid: str, high: str) -> str:
    if x < a:
        return low
    if x < b:
        return mid
    return high


def words(metrics: dict) -> dict:
    precip = lvl3(metrics["precipitation"], 0.25, 0.65, "низкая", "умеренная", "высокая")
    wind = lvl3(metrics["wind"], 0.25, 0.65, "слабый", "заметный", "порывистый")
    press = lvl3(metrics["pressure"], 0.25, 0.65, "спокойное", "переменное", "нестабильное")

    t = metrics["temperature"]
    temp = lvl3(t, 0.30, 0.75, "прохладная", "тёплая", "горячая")
    if t > 0.88:
        temp = "перегретая"

    conf = lvl3(metrics["confidence"], 0.35, 0.75, "низкая", "средняя", "высокая")
    return {"precip": precip, "wind": wind, "press": press, "temp": temp, "conf": conf}


# -----------------------------
# TEXT ENGINE (MORE VARIETY)
# -----------------------------
ANCHORS = [
    "☁️ Прогноз общественной погоды",
    "🌦 Политико-метеосводка",
    "⛅ Прогноз по атмосферным обсуждениям",
    "🌤 Городская погодная сводка по повестке",
    "🌥 Прогноз настроений и заголовков",
    "🛰 Сводка с радара ленты",
    "📡 Облачный бюллетень новостей",
]

VOICE_TAGS = [
    "Говорит метеостанция здравого смысла.",
    "На связи синоптики реальности.",
    "Передаём с фронта заголовков.",
    "Доклад с балкона критического мышления.",
    "Сводка с метеорадара комментариев.",
    "Служба наблюдения за повесткой сообщает.",
    "Официально-неофициальная метеослужба: внимание.",
]

OPENERS = [
    "За последние 24 часа воздух заметно наэлектризовался словами.",
    "За сутки в атмосфере накопилось достаточно шума, чтобы он начал казаться погодой.",
    "Последние 24 часа: лента ведёт себя как климат, но это всё ещё эмоции.",
    "Сутки были насыщены упоминаниями — местами с эффектом грома без дождя.",
    "Суточный прогноз: вероятность событий оценивается по публичным сигналам.",
]

MORNING_TEMPLATES = [
    "Утром вероятны {phenomenon} — явление {desc}.",
    "С утра возможны {phenomenon}: {desc}.",
    "Первая половина дня обещает {phenomenon}. По ощущениям — {desc}.",
    "На утреннем горизонте: {phenomenon}. Характер: {desc}.",
    "Утро приносит {phenomenon}, и это {desc}.",
]

PHENOMENA = [
    "локальные выступления и митинговая повестка",
    "точечные всплески уличной активности",
    "волны солидарности и встречные течения",
    "скопления людей вокруг громких тем",
    "порывы плакатов и лозунгов (местами)",
    "мелкая морось дискуссий вокруг выходов на улицу",
    "облачность из призывов и контрпризывов",
]

DESCS = [
    "шумное, но обычно кратковременное",
    "визуально плотное, но часто переменное",
    "эмоционально громкое, но не всегда длительное",
    "с оттенком «сейчас-сейчас» и быстрым рассеиванием",
    "похожее на грозу: много звука, мало осадков",
    "то сгущается, то исчезает — как будто само сомневается",
]

DAY_TEMPLATES = [
    "Днём ожидаются {day_event}; рекомендуется {advice}.",
    "После обеда возможны {day_event}. На всякий случай — {advice}.",
    "Во второй половине дня — {day_event}. Лучше держать рядом: {advice}.",
    "К середине дня поднимаются {day_event}. Практика дня: {advice}.",
    "Днём — {day_event}. Метеозащита: {advice}.",
    "Дневной фон: {day_event}. Совет: {advice}.",
]

DAY_EVENTS = [
    "порывы «breaking news»",
    "облака срочных интерпретаций",
    "резкие смены ветра в заголовках",
    "вспышки спорных тезисов",
    "ливни из «экспертных» выводов",
    "перепады тона в комментариях",
    "переохлаждение фактов и перегрев мнений",
    "кратковременные штормы в соцсетях",
]

ADVICES = [
    "зонт критического мышления",
    "куртку здравого смысла",
    "пауза между «увидел» и «поверил»",
    "проверка источников перед репостом",
    "ограничитель новостного скролла",
    "тёплый чай и холодная голова",
    "режим «не спорю на голодный мозг»",
    "правило двух вкладок: факт + первоисточник",
]

EVENING_TEMPLATES = [
    "К вечеру возможен {evening}: {evening_desc}.",
    "Ближе к вечеру — {evening}. Итог: {evening_desc}.",
    "Вечером приходит {evening} — и {evening_desc}.",
    "К ночи вероятен {evening}. Обычно это когда {evening_desc}.",
    "Финал дня: {evening}. Это значит — {evening_desc}.",
]

EVENINGS = [
    "шаббат-бриз",
    "затишье вне ленты",
    "режим «отложенные новости»",
    "возврат к человеческому масштабу",
    "тихая пауза в споре",
    "проветривание головы от новостей",
    "вечерняя тишина без срочности",
]

EVENING_DESCS = [
    "шум стихает, а смысл становится слышнее",
    "темп падает, и хочется говорить тише",
    "появляется шанс на нормальные слова",
    "вдруг оказывается, что люди важнее дискуссий",
    "заголовки откладываются, а жизнь остаётся",
    "вопросы остаются, но крик уходит",
    "можно зажечь свет — и не доказывать его необходимость",
]

METRIC_TEMPLATES = [
    "🌡 Температура общественного мнения — **{temp}**.\n🌬 Ветер заголовков — **{wind}**.\n🌍 Международное давление — **{press}**.",
    "🌡 Температура: **{temp}** | 🌬 Ветер: **{wind}** | 🌍 Давление: **{press}**.",
    "🌡 По ощущениям: **{temp}**.\n🌬 Порывы: **{wind}**.\n🌍 Давление: **{press}**.",
    "🌡 Состояние воздуха: **{temp}**.\n🌬 Движение заголовков: **{wind}**.\n🌍 Давление внешнее: **{press}**.",
]

RADAR_HEADERS = [
    "📡 Радар за 24 часа",
    "🛰 Радар суток",
    "📍 Суточный радар упоминаний",
    "🧭 Показания за сутки",
    "🗞 Индекс ленты за 24ч",
]

RADAR_LINES = [
    "Публичных сигналов за 24 часа: **{n}**.",
    "За сутки найдено упоминаний/анонсов: **{n}**.",
    "Суточное количество сигналов: **{n}**.",
    "Индекс упоминаний (24ч): **{n}**.",
    "Сводка счётчика за 24 часа: **{n}**.",
]

CONF_TEMPLATES = [
    "🔎 Уверенность прогноза: **{conf}** (больше сигналов → выше уверенность).",
    "🔎 Надёжность оценки: **{conf}**.",
    "🔎 Доверие к прогнозу: **{conf}**.",
    "🔎 Качество сигнала: **{conf}**.",
]

ASIDES = [
    "🧲 Магнитных бурь не ожидается, но внутренние — возможны.",
    "🪟 Рекомендуется проветрить ленту и закрыть вкладки со слухами.",
    "🧊 Осторожно: лёд в комментариях. Держитесь ближе к фактам.",
    "🧯 При перегреве — выключить спор и включить дыхание.",
    "🧠 Побочный эффект новостей: уверенность без доказательств.",
    "🧾 Если кто-то кричит «всё очевидно» — проверьте, что именно.",
    "🧘 Минимум: не спорить в моменте. Максимум: быть человеком.",
]

FINALS = [
    "Береги себя: даже пасмурная повестка не отменяет свет.",
    "Помни: свет не требует разрешения у заголовков.",
    "Береги голову и сердце: в любую погоду можно оставаться человеком.",
    "Даже если небо спорит — свеча всё равно горит.",
    "Погода меняется. Человечность — тоже может, если её тренировать.",
    "Если стало шумно — сделай тише внутри. Это тоже навык.",
    "Не отменяй свет из-за прогноза. Добавь его сам.",
]

MICRO_SECTIONS = [
    "🧾 Местами возможна путаница между «анонсом» и «обсуждением». Это нормально: погода слов — коварна.",
    "🧩 Иногда заголовок — это облако без дождя. Не выдавайте его за климат.",
    "🧷 Короткое правило: один факт — два источника.",
    "🕯️ Если день тяжёлый — уменьши скорость. Это не капитуляция, это управление.",
    "🧠 Напоминание: громкость — не аргумент.",
]

MODELS = ["classic", "philosophical", "dry", "poetic", "radio", "minimal"]


def build_message(city: str, metrics: dict, top_articles: List[Article]) -> str:
    w = words(metrics)
    mode = random.choices(MODELS, weights=[0.35, 0.18, 0.12, 0.10, 0.15, 0.10], k=1)[0]

    title = f"{pick(ANCHORS)}: {city}"
    voice = pick(VOICE_TAGS)
    opener = pick(OPENERS)

    morning = pick(MORNING_TEMPLATES).format(phenomenon=pick(PHENOMENA), desc=pick(DESCS))
    day = pick(DAY_TEMPLATES).format(day_event=pick(DAY_EVENTS), advice=pick(ADVICES))
    evening = pick(EVENING_TEMPLATES).format(evening=pick(EVENINGS), evening_desc=pick(EVENING_DESCS))

    metrics_block = pick(METRIC_TEMPLATES).format(temp=w["temp"], wind=w["wind"], press=w["press"])
    radar_block = f"{pick(RADAR_HEADERS)}\n" + pick(RADAR_LINES).format(n=int(metrics["n"]))
    conf_block = pick(CONF_TEMPLATES).format(conf=w["conf"])
    final = pick(FINALS)

    # Конструктор секций: очень вариативный порядок
    sections = []

    if mode == "minimal":
        # коротко, но не одинаково
        sections.append(title)
        sections.append(radar_block)
        if maybe(0.7):
            sections.append(metrics_block)
        sections.append(final)
    else:
        # верх
        if mode == "radio":
            sections.append(f"📻 {title}")
            sections.append(voice)
            if maybe(0.75):
                sections.append(opener)
        elif mode == "dry":
            sections.append(title)
            sections.append("Сводка за сутки по публичным сигналам.")
        elif mode == "poetic":
            sections.append(f"{title}\n{voice}\nСегодня воздух пахнет словами.")
        elif mode == "philosophical":
            sections.append(f"{title}\n{voice}\nГлавное — не путать громкость с правдой.")
        else:
            sections.append(f"{title}\n{voice}")
            if maybe(0.6):
                sections.append(opener)

        # середина (утро/день/вечер), иногда меняем порядок
        trio = [morning, day, evening]
        if maybe(0.35):
            random.shuffle(trio)
        # часто оставляем вечер в конце (чтобы был “моральный выход”)
        if maybe(0.75):
            trio = [x for x in trio if x != evening] + [evening]
        sections.extend(trio)

        # метрики/радар/уверенность — в разном порядке
        tail = [metrics_block, radar_block]
        if maybe(0.85):
            tail.append(conf_block)
        random.shuffle(tail)
        sections.extend(tail)

        # вставки-пасхалки (может быть 0..2)
        if maybe(0.35):
            sections.append(pick(ASIDES))
        if maybe(0.25):
            sections.append(pick(MICRO_SECTIONS))

        # финал
        sections.append(final)

    # ссылки на источники — иногда показываем, иногда нет
    if top_articles and maybe(0.70):
        lines = []
        for a in top_articles[:6]:
            lines.append(f"• {a.title}\n  {a.url}")
        sections.append("📰 Открытые сигналы (последние 24ч):\n" + "\n".join(lines))

    text = "\n\n".join(sections).strip()

    # лёгкие “мутации” текста, чтобы ещё меньше повторов
    if maybe(0.25):
        text = text.replace("за 24 часа", "за последние сутки").replace("за сутки", "за 24 часа")
    if maybe(0.18):
        text = text.replace("общественной", "публичной")
    return text


# -----------------------------
# ROUTES
# -----------------------------
@router.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer(
        "Формат:\n"
        "• /forecast Tallinn\n"
        "• /forecast London, UK\n\n"
        "Я анализирую публичные упоминания/анонсы за **последние 24 часа** и выдаю «погодную» сводку."
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
        try:
            articles_24h = await gdelt_fetch_articles(session, city=city)
        except Exception as e:
            logger.warning("GDELT fetch failed: %r", e)
            articles_24h = []

    metrics = compute_metrics(articles_24h)
    top_articles = sorted(articles_24h, key=lambda a: a.seendate, reverse=True)

    await message.answer(build_message(city, metrics, top_articles), disable_web_page_preview=True)


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
