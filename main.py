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
from openai import OpenAI

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

SYSTEM_PROMPT = """Ты – юридический ассистент. Проанализируй изображение страницы судебного решения. Извлеки структурированные данные согласно JSON-схеме. Требования:
- ФИО: только в именительном падеже, полностью (Иванов Иван Иванович), без инициалов. Применяется к judge, plaintiff, defendant, third_party и people[].full_name.
- role: строго из словаря: истец, ответчик, третье лицо, судья, представитель, секретарь.
- legal_basis: формат "ст. N ГК РФ" или "ст. N АПК РФ" (и др.), без дублей.
- key_findings: каждый вывод в виде {"finding": "текст", "evidence": "короткая цитата из решения"}. evidence обязателен для подтверждения вывода; если цитаты нет — пустая строка.
Верни только JSON, без пояснений.

{
  "document_type": "...",
  "court_name": "...",
  "case_number": "...",
  "date": "ДД-ММ-ГГГГ",
  "judge": "...",
  "plaintiff": "...",
  "defendant": "...",
  "third_party": "...",
  "claim_amount": {"value": число, "currency": "руб."},
  "claim_subject": "...",
  "ruling": "...",
  "legal_basis": ["ст. 1 ГК РФ", "ст. 2 АПК РФ"],
  "summary": "Краткое резюме содержания страницы (если на странице нет существенной информации, оставь пустую строку)",
  "key_findings": [
    {"finding": "Суд отказал в иске", "evidence": "в связи с пропуском срока исковой давности, стр. 5"},
    ...
  ],
  "people": [
    {"full_name": "Иванов Иван Иванович", "role": "истец"},
    ...
    ],
  "dates": [{"label": "дата договора", "value": "01-01-2023"}],
  "amounts": [{"label": "цена иска", "value": 100000, "currency": "руб."}],
  "tables": [{"name": "...", "columns": ["col1"], "rows": [["val1"]]}]
}

Обрати внимание: summary и key_findings формируются для текущей страницы; при объединении они будут агрегированы. Поля summary и key_findings должны присутствовать всегда (хотя бы пустой массив для key_findings и пустая строка для summary)."""


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

    completion = client.chat.completions.create(
        model=model,
        temperature=0.1,
        response_format={"type": "json_object"},
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

    content = completion.choices[0].message.content.strip()

    # Убрать возможные markdown-обёртки
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
        )

    # Гарантируем наличие summary и key_findings
    if "summary" not in data:
        data["summary"] = ""
    if "key_findings" not in data:
        data["key_findings"] = []

    return data


def merge_results(partial_results: list[dict]) -> dict:
    """
    Объединяет частичные JSON-результаты по страницам в один итоговый документ.
    - Уникальные поля (court_name, case_number и т.д.): первое непустое значение.
    - Списки (people, dates, amounts, tables, key_findings): конкатенация.
    - summary: самое длинное непустое резюме (или первое непустое).
    """
    if not partial_results:
        return {
            "summary": "",
            "key_findings": [],
        }

    # Поля, для которых берём первое непустое значение
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
        "ruling",
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

    # summary: самое длинное непустое (или первое непустое)
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
    """Нормализует статью к виду «ст. N ГК РФ» / «ст. N АПК РФ» и т.д."""
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None
    # Приводим к единому формату: ст. N <КОДЕКС> РФ
    m = re.search(r"ст\.?\s*(\d+(?:\s*/\s*\d+)?)\s*(.+)?", s, re.I)
    if m:
        num, code = m.group(1), (m.group(2) or "").strip()
        code = code or "ГК РФ"
        if not code.upper().endswith("РФ"):
            code = f"{code} РФ" if code else "ГК РФ"
        return f"ст. {num} {code}"
    return s if s.startswith("ст.") or "ст." in s else None


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
    """Проверяет и нормализует дату к ГГГГ-ММ-ДД или ДД.ММ.ГГГГ."""
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    # ДД.ММ.ГГГГ, ДД-ММ-ГГГГ
    for fmt in ("%d.%m.%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            dt = datetime.strptime(re.sub(r"\s+", "", s), fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return s


def postprocess(data: dict) -> dict:
    """
    Постобработка: дедупликация people/legal_basis, валидация чисел и дат.
    """
    # Дедупликация people по (full_name, role)
    if "people" in data and isinstance(data["people"], list):
        seen: set[tuple[str, str]] = set()
        out = []
        for p in data["people"]:
            if not isinstance(p, dict):
                continue
            name = (p.get("full_name") or "").strip()
            role = (p.get("role") or "").strip().lower()
            if role not in VALID_ROLES:
                role = "представитель" if role else ""
            if not name:
                continue
            key = (name, role)
            if key in seen:
                continue
            seen.add(key)
            out.append({"full_name": name, "role": role or "представитель"})
        data["people"] = out

    # Дедупликация legal_basis, нормализация формата
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

    # Валидация dates
    if "date" in data and data["date"]:
        data["date"] = _parse_date(data["date"]) or data["date"]
    if "dates" in data and isinstance(data["dates"], list):
        out_dates = []
        for d in data["dates"]:
            if not isinstance(d, dict):
                continue
            val = d.get("value")
            parsed = _parse_date(val)
            if parsed or val:
                out_dates.append({
                    "label": (d.get("label") or "").strip() or "дата",
                    "value": parsed or str(val),
                })
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
