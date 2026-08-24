"""
Инструмент для извлечения структурированных данных из судебных решений
(скан-образы, PDF) с использованием OpenAI GPT-4o.
"""

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import tempfile
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from openai import APIError, OpenAI, OpenAIError

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

load_dotenv()

# Расширения изображений
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
# Расширение PDF
PDF_EXTENSION = ".pdf"

VALID_ROLES = frozenset({
    "истец", "ответчик", "третье лицо", "судья", "представитель", "секретарь",
})

# JSON Schema for Chat Completions Structured Outputs (strict).
# Optional scalars are nullable; all object properties are required.
EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "document_type": {"type": ["string", "null"]},
        "court_name": {"type": ["string", "null"]},
        "case_number": {"type": ["string", "null"]},
        "date": {
            "type": ["string", "null"],
            "description": "Дата решения в формате YYYY-MM-DD, если надёжно извлечена",
        },
        "judge": {"type": ["string", "null"]},
        "plaintiff": {"type": ["string", "null"]},
        "defendant": {"type": ["string", "null"]},
        "third_party": {"type": ["string", "null"]},
        "claim_amount": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "properties": {
                "value": {"type": ["number", "null"]},
                "currency": {"type": ["string", "null"]},
            },
            "required": ["value", "currency"],
        },
        "claim_subject": {"type": ["string", "null"]},
        "ruling": {"type": ["string", "null"]},
        "legal_basis": {
            "type": "array",
            "items": {"type": "string"},
        },
        "summary": {
            "type": "string",
            "description": "Краткое резюме содержания текущей страницы; пустая строка, если нет существенной информации",
        },
        "key_findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "finding": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["finding", "evidence"],
            },
        },
        "people": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "full_name": {"type": "string"},
                    "role": {"type": "string"},
                },
                "required": ["full_name", "role"],
            },
        },
        "dates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "label": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["label", "value"],
            },
        },
        "amounts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "label": {"type": "string"},
                    "value": {"type": "number"},
                    "currency": {"type": "string"},
                },
                "required": ["label", "value", "currency"],
            },
        },
        "tables": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "rows": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
                "required": ["name", "columns", "rows"],
            },
        },
    },
    "required": [
        "document_type",
        "court_name",
        "case_number",
        "date",
        "judge",
        "plaintiff",
        "defendant",
        "third_party",
        "claim_amount",
        "claim_subject",
        "ruling",
        "legal_basis",
        "summary",
        "key_findings",
        "people",
        "dates",
        "amounts",
        "tables",
    ],
}

RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "court_decision_extraction",
        "strict": True,
        "schema": EXTRACTION_SCHEMA,
    },
}

SYSTEM_PROMPT = """Ты – юридический ассистент. Проанализируй изображение страницы судебного решения. Извлеки структурированные данные строго по JSON-схеме ответа. Требования:
- ФИО: только в именительном падеже, полностью (Иванов Иван Иванович), без инициалов. Применяется к judge, plaintiff, defendant, third_party и people[].full_name.
- role: предпочтительно из словаря: истец, ответчик, третье лицо, судья, представитель, секретарь. Если роль на странице иная — укажи её как есть; не подменяй неизвестную роль на «представитель». Если роль неизвестна — оставь пустую строку.
- date и dates[].value: формат YYYY-MM-DD, только если дата надёжно читается; иначе date = null / не выдумывай дату.
- legal_basis: указывай кодекс (ГК РФ, АПК РФ и т.п.) только если он явно виден или однозначно следует из текста на этой странице. Для голой ссылки вида «ст. N» без кодекса НЕ угадывай кодекс — верни «ст. N» как есть. Не добавляй правовую информацию, которой нет на странице. Без дублей.
- key_findings: каждый вывод в виде {"finding": "текст", "evidence": "короткая цитата из решения"}. evidence обязателен для подтверждения вывода; если цитаты нет — пустая строка.
- summary: краткое резюме содержания ТЕКУЩЕЙ страницы; если на странице нет существенной информации — пустая строка.
Верни только JSON по схеме, без пояснений."""


def encode_image_to_data_url(image_path: str) -> str:
    """Кодирует изображение в data URL для API."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Файл не найден: {image_path}")

    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/jpeg"

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{b64}"


def analyze_image(image_path: str, model: str = "gpt-4o") -> dict:
    """
    Анализирует изображение страницы судебного решения и возвращает структурированный JSON.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Не задан OPENAI_API_KEY. Укажите ключ в переменных окружения или в файле .env"
        )

    client = OpenAI(api_key=api_key)
    image_data_url = encode_image_to_data_url(image_path)

    try:
        completion = client.chat.completions.create(
            model=model,
            temperature=0.1,
            response_format=RESPONSE_FORMAT,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Проанализируй изображение страницы судебного решения. "
                                "Извлеки структурированные данные по схеме. Верни только JSON."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url},
                        },
                    ],
                },
            ],
        )
    except APIError as e:
        raise RuntimeError(f"Ошибка OpenAI API: {e}") from e
    except OpenAIError as e:
        raise RuntimeError(f"Ошибка клиента OpenAI: {e}") from e

    if not completion.choices:
        raise RuntimeError("Пустой ответ OpenAI API: нет choices в completion.")

    message = completion.choices[0].message
    refusal = getattr(message, "refusal", None)
    if refusal:
        raise RuntimeError(f"Модель отказалась обработать запрос: {refusal}")

    content = message.content
    if content is None or not str(content).strip():
        raise RuntimeError("Пустой ответ модели: отсутствует содержимое сообщения.")

    content = str(content).strip()

    # Убрать возможные markdown-обёртки (на случай нестандартного ответа)
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        )

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Не удалось распарсить ответ модели как JSON: {e}\nОтвет:\n{content[:500]}..."
        ) from e

    if not isinstance(data, dict):
        raise ValueError(
            f"Ожидался JSON-объект, получен {type(data).__name__}."
        )

    # Гарантируем наличие summary и key_findings
    if "summary" not in data or data["summary"] is None:
        data["summary"] = ""
    if "key_findings" not in data or data["key_findings"] is None:
        data["key_findings"] = []

    return data


def merge_results(partial_results: list[dict]) -> dict:
    """
    Объединяет частичные JSON-результаты по страницам в один итоговый документ.
    - Уникальные поля (court_name, case_number и т.д.): первое непустое значение.
    - ruling: последнее непустое значение (резолютивная часть обычно ближе к концу).
    - Списки (people, dates, amounts, tables, key_findings): конкатенация.
    - summary: самое длинное непустое резюме страницы (не синтез по всему документу).
    """
    if not partial_results:
        return {
            "summary": "",
            "key_findings": [],
        }

    # Поля, для которых берём первое непустое значение (кроме ruling)
    single_value_fields = [
        "document_type",
        "court_name",
        "case_number",
        "date",
        "judge",
        "plaintiff",
        "defendant",
        "third_party",
        "claim_subject",
    ]

    # Поля-списки: конкатенируем
    list_fields = ["people", "dates", "amounts", "tables", "key_findings", "legal_basis"]

    merged = {}

    for key in single_value_fields:
        for part in partial_results:
            val = part.get(key)
            if val is not None and (val != "" if isinstance(val, str) else True):
                merged[key] = val
                break

    # ruling: последнее непустое (пустые поздние значения не затирают ранее найденное)
    last_ruling = None
    for part in partial_results:
        val = part.get("ruling")
        if val is not None and (val != "" if isinstance(val, str) else True):
            last_ruling = val
    if last_ruling is not None:
        merged["ruling"] = last_ruling

    # claim_amount — одно значение, первое непустое
    for part in partial_results:
        amt = part.get("claim_amount")
        if amt and isinstance(amt, dict) and amt.get("value") is not None:
            merged["claim_amount"] = amt
            break

    for key in list_fields:
        combined = []
        for part in partial_results:
            items = part.get(key)
            if items is None:
                continue
            if isinstance(items, list):
                for item in items:
                    if key == "key_findings" and isinstance(item, str):
                        combined.append({"finding": item, "evidence": ""})
                    elif key == "key_findings" and isinstance(item, dict):
                        combined.append({
                            "finding": item.get("finding", ""),
                            "evidence": item.get("evidence", ""),
                        })
                    else:
                        combined.append(item)
            elif isinstance(items, str) and key == "legal_basis":
                combined.append(items)
        if combined:
            merged[key] = combined

    # summary: самое длинное непустое резюме страницы (не синтез всего документа)
    summaries = [p.get("summary", "") or "" for p in partial_results]
    non_empty = [s for s in summaries if s.strip()]
    if non_empty:
        merged["summary"] = max(non_empty, key=len)
    else:
        merged["summary"] = summaries[0] if summaries else ""

    # Гарантируем key_findings
    if "key_findings" not in merged:
        merged["key_findings"] = []

    return merged


def _normalize_legal_basis(s: str) -> str | None:
    """Нормализует пробелы/формат «ст. N …»; не добавляет кодекс, если его не было."""
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None
    m = re.search(r"ст\.?\s*(\d+(?:\s*/\s*\d+)?)\s*(.*)?", s, re.I)
    if m:
        num = re.sub(r"\s+", "", m.group(1))
        rest = re.sub(r"\s+", " ", (m.group(2) or "").strip())
        if rest:
            return f"ст. {num} {rest}"
        return f"ст. {num}"
    cleaned = re.sub(r"\s+", " ", s)
    if cleaned.lower().startswith("ст.") or "ст." in cleaned.lower():
        return cleaned
    return None


def _parse_number(val: Any) -> float | int | None:
    """Извлекает число из value (может быть строкой с пробелами)."""
    if val is None:
        return None
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        return int(val) if val == int(val) else float(val)
    if isinstance(val, str):
        cleaned = re.sub(r"[\s\xa0]", "", val).replace(",", ".")
        try:
            f = float(cleaned)
            return int(f) if f == int(f) else f
        except ValueError:
            pass
    return None


def _parse_date(val: Any) -> str | None:
    """Нормализует дату к YYYY-MM-DD. Непарсабельные значения → None (без выдумывания)."""
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    for fmt in ("%d.%m.%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            dt = datetime.strptime(re.sub(r"\s+", "", s), fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def postprocess(data: dict) -> dict:
    """
    Постобработка: дедупликация people/legal_basis, валидация чисел и дат.
    Не выводит кодексы и роли, которых не было в данных модели.
    """
    # Дедупликация people по (full_name, role)
    if "people" in data and isinstance(data["people"], list):
        seen: set[tuple[str, str]] = set()
        out = []
        for p in data["people"]:
            if not isinstance(p, dict):
                continue
            name = (p.get("full_name") or "").strip()
            raw_role = p.get("role")
            if raw_role is None:
                role = ""
            else:
                role = str(raw_role).strip()
            role_key = role.lower()
            if role_key in VALID_ROLES:
                role = role_key
            # unknown non-empty role: preserve as-is; empty stays empty
            if not name:
                continue
            key = (name, role)
            if key in seen:
                continue
            seen.add(key)
            out.append({"full_name": name, "role": role})
        data["people"] = out

    # Дедупликация legal_basis, нормализация формата без дописывания кодекса
    if "legal_basis" in data and isinstance(data["legal_basis"], list):
        seen_lb: set[str] = set()
        out_lb = []
        for item in data["legal_basis"]:
            s = item if isinstance(item, str) else str(item)
            norm = _normalize_legal_basis(s)
            if norm and norm not in seen_lb:
                seen_lb.add(norm)
                out_lb.append(norm)
        data["legal_basis"] = out_lb

    # Валидация claim_amount
    if "claim_amount" in data and isinstance(data["claim_amount"], dict):
        ca = data["claim_amount"]
        v = _parse_number(ca.get("value"))
        if v is not None and v >= 0:
            ca["value"] = int(v) if v == int(v) else v
            ca["currency"] = ca.get("currency") or "руб."
        elif v is not None and v < 0:
            ca["value"] = int(v) if v == int(v) else v

    # Валидация amounts
    if "amounts" in data and isinstance(data["amounts"], list):
        out_amt = []
        for a in data["amounts"]:
            if not isinstance(a, dict):
                continue
            v = _parse_number(a.get("value"))
            if v is not None:
                out_amt.append({
                    "label": (a.get("label") or "").strip() or "сумма",
                    "value": int(v) if v == int(v) else v,
                    "currency": a.get("currency") or "руб.",
                })
        data["amounts"] = out_amt

    # Валидация dates: только надёжно распознанные → YYYY-MM-DD
    if "date" in data and data["date"]:
        data["date"] = _parse_date(data["date"])
    if "dates" in data and isinstance(data["dates"], list):
        out_dates = []
        for d in data["dates"]:
            if not isinstance(d, dict):
                continue
            val = d.get("value")
            parsed = _parse_date(val)
            if parsed:
                out_dates.append({
                    "label": (d.get("label") or "").strip() or "дата",
                    "value": parsed,
                })
            # непарсабельные даты не включаем и не выдумываем
        data["dates"] = out_dates

    # Нормализация key_findings к {finding, evidence}
    if "key_findings" in data and isinstance(data["key_findings"], list):
        out_kf = []
        for kf in data["key_findings"]:
            if isinstance(kf, str):
                out_kf.append({"finding": kf, "evidence": ""})
            elif isinstance(kf, dict):
                f = (kf.get("finding") or "").strip()
                if f:
                    out_kf.append({
                        "finding": f,
                        "evidence": (kf.get("evidence") or "").strip(),
                    })
        data["key_findings"] = out_kf

    return data


def get_file_extension(path: str) -> str:
    return os.path.splitext(path)[1].lower()


def pdf_to_image_paths(pdf_path: str, all_pages: bool) -> list[str]:
    """
    Конвертирует PDF в список путей к временным изображениям.
    all_pages=True — все страницы (по умолчанию), False — только первая.
    """
    if convert_from_path is None:
        raise RuntimeError(
            "Для работы с PDF установите pdf2image: pip install pdf2image. "
            "Также нужен Poppler (https://github.com/oschwartz10612/poppler-windows/releases)."
        )

    # Путь к Poppler: переменная POPPLER_PATH или в PATH
    poppler_path = os.getenv("POPPLER_PATH")
    kwargs = {"poppler_path": poppler_path} if poppler_path else {}

    try:
        if all_pages:
            pages = convert_from_path(pdf_path, **kwargs)
        else:
            pages = convert_from_path(pdf_path, first_page=1, last_page=1, **kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Ошибка конвертации PDF (проверьте наличие Poppler): {e}\n"
            "Windows: скачайте Poppler с https://github.com/oschwartz10612/poppler-windows/releases, "
            "распакуйте и добавьте папку bin в PATH или укажите POPPLER_PATH в .env (путь к папке bin)."
        ) from e

    temp_paths = []
    try:
        for page in pages:
            fd, path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            page.save(path, "PNG")
            temp_paths.append(path)
        return temp_paths
    except Exception:
        for p in temp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass
        raise


def normalize_path(path: str) -> str:
    """Разворачивает ~ и относительный путь в абсолютный."""
    path = os.path.expanduser(path.strip())
    return os.path.abspath(path)


def run(
    file_path: str,
    *,
    model: str = "gpt-4o",
    all_pages: bool = True,
) -> dict:
    """
    Обрабатывает файл (изображение или PDF) и возвращает объединённый JSON.
    Для PDF по умолчанию обрабатываются все страницы, затем результаты объединяются.
    """
    file_path = normalize_path(file_path)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"Файл не найден: {file_path}\n"
            "Проверьте путь. Скопируйте его из проводника (ПКМ по файлу → «Копировать как путь»)."
        )

    ext = get_file_extension(file_path)

    if ext == PDF_EXTENSION:
        image_paths = pdf_to_image_paths(file_path, all_pages)
        try:
            results = []
            for img_path in image_paths:
                results.append(analyze_image(img_path, model=model))
            merged = merge_results(results)
            return postprocess(merged)
        finally:
            for p in image_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass
    elif ext in IMAGE_EXTENSIONS:
        data = analyze_image(file_path, model=model)
        return postprocess(data)
    else:
        raise ValueError(
            f"Неподдерживаемый формат: {ext}. "
            f"Допустимы: {', '.join(IMAGE_EXTENSIONS | {PDF_EXTENSION})}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Извлечение структурированных данных из судебных решений (PDF, изображения) с помощью GPT-4o."
    )
    parser.add_argument(
        "path_to_file",
        help="Путь к PDF или изображению (.jpg, .jpeg, .png)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Файл для сохранения JSON (по умолчанию — вывод в stdout)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Модель OpenAI (по умолчанию: gpt-4o)",
    )
    parser.add_argument(
        "--first-page-only",
        action="store_true",
        help="Обрабатывать только первую страницу PDF (для экономии токенов)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Форматировать JSON с отступами",
    )

    args = parser.parse_args()

    try:
        data = run(
            args.path_to_file,
            model=args.model,
            all_pages=not args.first_page_only,
        )
    except FileNotFoundError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(1)

    json_str = (
        json.dumps(data, ensure_ascii=False, indent=2)
        if args.pretty
        else json.dumps(data, ensure_ascii=False)
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"Результат сохранён в {args.output}", file=sys.stderr)
    else:
        print(json_str)


if __name__ == "__main__":
    main()
