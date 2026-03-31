# %% [markdown]
# # DocVQA field labeling pipeline
#
# Что делает этот ноутбук:
# - размечает пары (question, answer) по типу поля
# - поддерживает train / validation
# - использует rule-based prelabeling + LLM fallback
# - сохраняет стабильный example_id
# - поддерживает checkpoint / resume
# - сохраняет статистики
# - поддерживает golden evaluation
#
# ВАЖНО:
# - rule-based логика и связанные patterns откатаны к старой версии пайплайна
# - общий пайплайн сохранён в более удобной расширенной форме

# %%
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import load_dataset
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm.auto import tqdm


# %%
# =========================
# Config
# =========================

PROJECT_ROOT = Path.cwd().resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"

DATASET_NAME = "pixparse/docvqa-single-page-questions"

# Какие сплиты прогонять
SPLITS = ["validation"]  # потом можно заменить на ["validation", "train"]

# Ограничения по сплитам: None = весь сплит
LIMITS = {
    "validation": None,
    "train": None,
}

MODEL = "GigaChat"
VERIFY_SSL_CERTS = False

# Частота сохранения результатов
SAVE_EVERY = 50

# Пауза между запросами
SLEEP_SEC = 0.2

# Для воспроизводимости
RANDOM_SEED = 42

# Если True — уже размеченные строки подхватываются из CSV и пропускаются
RESUME = False

# Если True — для уверенных rule-based случаев LLM не вызывается
USE_RULES_FIRST = True

# Порог уверенности rule-based для прямого принятия
RULE_ACCEPT_THRESHOLD = 0.93

# Если True — сохраняем сырой ответ модели
STORE_RAW_LLM_OUTPUT = True

# Оценивать golden
RUN_GOLD_EVAL = True

# Для каких сплитов искать gold-файлы
GOLD_EVAL_SPLITS = ["validation"]

FINAL_EXPORT_BASENAMES = {
    "validation": "pixparse__docvqa-single-page-questions__validation__field_labels_v1",
}

# Каталог артефактов
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "field_labeling"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Подкаталоги
RUNS_DIR = OUTPUT_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

MERGED_DIR = OUTPUT_DIR / "merged"
MERGED_DIR.mkdir(parents=True, exist_ok=True)

GOLD_DIR = OUTPUT_DIR / "gold_annotation"
GOLD_DIR.mkdir(parents=True, exist_ok=True)

EVAL_DIR = GOLD_DIR / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


# %%
# =========================
# Taxonomy
# =========================

ALLOWED_TYPES = {
    "PERSON_NAME",
    "ORG_NAME",
    "LOCATION",
    "ADDRESS",
    "DATE_TIME",
    "MONEY",
    "PERCENTAGE",
    "QUANTITY",
    "IDENTIFIER",
    "CONTACT",
    "DOCUMENT_REFERENCE",
    "TITLE_HEADER",
    "ROLE_TITLE",
    "CATEGORY_LABEL",
    "FREE_TEXT",
    "BOOLEAN",
    "OTHER",
}

SENSITIVITY_MAP = {
    "PERSON_NAME": "HIGH",
    "ADDRESS": "HIGH",
    "CONTACT": "HIGH",
    "IDENTIFIER": "HIGH",
    "ORG_NAME": "MEDIUM",
    "LOCATION": "MEDIUM",
    "DATE_TIME": "MEDIUM",
    "MONEY": "MEDIUM",
    "PERCENTAGE": "LOW",
    "QUANTITY": "LOW",
    "DOCUMENT_REFERENCE": "LOW",
    "TITLE_HEADER": "LOW",
    "ROLE_TITLE": "LOW",
    "CATEGORY_LABEL": "LOW",
    "FREE_TEXT": "LOW",
    "BOOLEAN": "LOW",
    "OTHER": "LOW",
}

GROUP_MAP = {
    "PERSON_NAME": "SENSITIVE_PII",
    "ADDRESS": "SENSITIVE_PII",
    "CONTACT": "SENSITIVE_PII",
    "IDENTIFIER": "SENSITIVE_PII",
    "ORG_NAME": "ENTITY",
    "LOCATION": "ENTITY",
    "ROLE_TITLE": "ENTITY",
    "DATE_TIME": "NUMERIC_FACTUAL",
    "MONEY": "NUMERIC_FACTUAL",
    "PERCENTAGE": "NUMERIC_FACTUAL",
    "QUANTITY": "NUMERIC_FACTUAL",
    "DOCUMENT_REFERENCE": "STRUCTURAL",
    "TITLE_HEADER": "STRUCTURAL",
    "CATEGORY_LABEL": "STRUCTURAL",
    "FREE_TEXT": "OPEN_TEXT",
    "BOOLEAN": "OPEN_TEXT",
    "OTHER": "OPEN_TEXT",
}

PROMPT_TEMPLATE = """Ты помогаешь размечать пары (вопрос, ответ) из датасета документов по типу поля.

Нужно выбрать РОВНО ОДИН тип из списка:

- PERSON_NAME — имя или ФИО человека.
- ORG_NAME — название организации, компании, банка, университета, учреждения.
- LOCATION — город, страна или географическое место.
- ADDRESS — почтовый адрес или его часть.
- DATE_TIME — дата, время, год, месяц, интервал дат.
- MONEY — денежная сумма или числовое значение финансового поля (даже если символ валюты отсутствует).
- PERCENTAGE — процент или доля в процентах.
- QUANTITY — числовое значение, не являющееся денежной суммой, процентом или идентификатором.
- IDENTIFIER — код, номер, account number, invoice number, id, reference code, номер записи.
- CONTACT — телефон, факс, email, website.
- DOCUMENT_REFERENCE — номер страницы, рисунка, таблицы, приложения, раздела или другой структурный указатель документа.
- TITLE_HEADER — заголовок документа, таблицы, графика, раздела, подпись, название документа или пункта списка/темы.
- ROLE_TITLE — должность, роль, титул, статус.
- CATEGORY_LABEL — короткая категориальная метка, тип объекта, тип документа, тип услуги, тип сущности, расшифровка аббревиатуры.
- FREE_TEXT — фраза или предложение, не относящееся к перечисленным типам.
- BOOLEAN — ответ да/нет, true/false, наличие/отсутствие.
- OTHER — только если ни один тип не подходит.

Правила:
1. Нельзя придумывать новые типы.
2. Отвечай только одним словом — названием типа.
3. Если это обычное число без валюты и без признаков идентификатора, выбирай QUANTITY.
4. Если это финансовое поле (expenses, sales, earnings, income, tax, budget, cost и т.п.), выбирай MONEY, даже если ответ выглядит как просто число.
5. Если это номер страницы/таблицы/рисунка/приложения, выбирай DOCUMENT_REFERENCE.
6. Если это имя человека, выбирай PERSON_NAME.
7. Если это заголовок или название раздела/таблицы/графика/документа/темы списка, выбирай TITLE_HEADER.
8. Если это короткий тип, вид, категория, сервис, расшифровка аббревиатуры — выбирай CATEGORY_LABEL.
9. Если ответ смешанный, выбирай тип главного целевого поля, которое спрашивается в вопросе.

Вопрос: {question}
Ответ: {answer}
"""


# %%
# =========================
# Regex / heuristics
# =========================

MONEY_RE = re.compile(
    r"""(?ix)
    ^
    \s*
    (
        [\$€£¥₹₽]\s*[\d,]+(?:\.\d+)? |
        [\d,]+(?:\.\d+)?\s*(usd|eur|gbp|rub|rs\.?|million|billion|thousand|/hour)
    )
    \s*$
    """
)

PERCENT_RE = re.compile(
    r"""(?ix)
    ^
    \s*
    [+-]?\d+(?:[.,]\d+)?\s*%
    \s*$
    """
)

PHONE_RE = re.compile(
    r"""(?x)
    ^
    \s*
    (?:\+?\d[\d\-\(\)\s]{5,}\d|\d{3,5})
    \s*$
    """
)

EMAIL_RE = re.compile(
    r"""(?ix)
    ^
    [A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}
    $
    """
)

URL_RE = re.compile(
    r"""(?ix)
    ^
    (https?://)?(www\.)?[a-z0-9\-]+\.[a-z]{2,}(/[^\s]*)?
    $
    """
)

DATE_LIKE_RE = re.compile(
    r"""(?ix)
    ^
    \s*
    (
        \d{1,2}[/-]\d{1,2}[/-]\d{2,4} |
        \d{4} |
        (jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]* |
        (jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(,\s*\d{2,4})? |
        \d{1,2}\s*(a\.?m\.?|p\.?m\.?) |
        \d{1,2}:\d{2}\s*(a\.?m\.?|p\.?m\.?)? |
        \d{4}\s*[~\-–]\s*\d{4}
    )
    \s*$
    """
)

NUMERIC_RE = re.compile(
    r"""(?x)
    ^
    \s*
    [+-]?\d+(?:[.,]\d+)?
    \s*$
    """
)

IDENTIFIER_RE = re.compile(
    r"""(?ix)
    ^
    \s*
    (?=.*[\d])
    [A-Z0-9][A-Z0-9\-/ ]{1,}
    \s*$
    """
)

ZIP_RE = re.compile(r"^\s*\d{4,10}(?:-\d{4})?\s*$")

BOOLEAN_VALUES = {"yes", "no", "true", "false", "y", "n"}

QUESTION_PATTERNS = {
    "DOCUMENT_REFERENCE": [
        r"\bpage number\b",
        r"\bpage no\b",
        r"\bpage mentioned\b",
        r"\bfigure number\b",
        r"\btable number\b",
        r"\bannex\b",
        r"\bsection number\b",
        r"\bfootnote number\b",
    ],
    "LOCATION": [
        r"\bwhich country\b",
        r"\bfrom which state\b",
        r"\bwhere are .* from\b",
        r"\blocated in which country\b",
        r"\bwhich state\b",
        r"\bwhich city\b",
        r"\bclinical center\b",
        r"\bwhich center\b",
        r"\bdealer is located\b",
    ],
    "CONTACT": [
        r"\bphone\b",
        r"\btelephone\b",
        r"\bfax\b",
        r"\bemail\b",
        r"\bwebsite\b",
        r"\bvoice mail\b",
    ],
    "ROLE_TITLE": [
        r"\brole\b",
        r"\bposition\b",
        r"\bchairman\b",
        r"\bpresident\b",
        r"\bsecretary\b",
        r"\bconsultant\b",
        r"\bceo\b",
        r"\bmanager\b",
        r"\bvice president\b",
        r"\bdirector\b",
    ],
    "PERSON_NAME": [
        r"^\s*who\b",
        r"\bwhose name\b",
        r"\bname of a person\b",
        r"\bclient\b",
        r"\bsender\b",
    ],
    "ORG_NAME": [
        r"\bname of the company\b",
        r"\bname of the bank\b",
        r"\bname of the university\b",
        r"\bname of the society\b",
        r"\bname of the airline\b",
        r"\bname of the consulting agency\b",
        r"\badvertising agency\b",
        r"\bchain corporate\b",
        r"\bvenue name\b",
        r"\bfederation\b",
        r"\bhead quartered\b",
        r"\bheadquartered\b",
        r"\borganization\b",
    ],
    "TITLE_HEADER": [
        r"\bheading\b",
        r"\btitle of the document\b",
        r"\btitle of the given\b",
        r"\btitle of the graph\b",
        r"\btitle of the bar graph\b",
        r"\bheading of the table\b",
        r"\bsubheading\b",
        r"\bwhat is printed below the logo\b",
        r"\bwhat is this notice about\b",
        r"\bwhat type of report is this\b",
        r"\bjournal of publication\b",
        r"\bfirst hot issue\b",
        r"\bsubject of this correspondence\b",
        r"\bwhat is ['‘\"]?table \d+['’\"]?\b",
        r"\bwhat is the footnote of\b",
        r"\by-axis indicate\b",
        r"\bx-axis\b",
        r"\btitle in the first rectangle\b",
        r"\btitle in the last rectangle\b",
    ],
    "ADDRESS": [
        r"\baddress\b",
        r"\bstreet\b",
        r"\bmailing address\b",
        r"\bzip ?code\b",
        r"\bzipcode\b",
    ],
    "DATE_TIME": [
        r"\bwhat date\b",
        r"\bwhen\b",
        r"\bwhat year\b",
        r"\bwhat month\b",
        r"\btime\b",
        r"\beffective date\b",
        r"\bdate mentioned\b",
        r"\bdate on\b",
    ],
    "MONEY": [
        r"\bexpense\b",
        r"\bexpenses\b",
        r"\bsalary\b",
        r"\bsalaries\b",
        r"\bamount paid\b",
        r"\bfare amount\b",
        r"\btax amount\b",
        r"\btotal including tax\b",
        r"\bsales\b",
        r"\brevenue\b",
        r"\bincome\b",
        r"\bearn(?:ing|ings)?\b",
        r"\bgross profit\b",
        r"\bnet profit\b",
        r"\bbudget\b",
        r"\bbudgeted\b",
        r"\bactual salaries\b",
        r"\bamount incurred\b",
        r"\bcost\b",
        r"\btotal intrinsic value\b",
        r"\bpbt\b",
        r"\bpat\b",
    ],
    "PERCENTAGE": [
        r"\bpercentage\b",
        r"\bshare\b",
        r"\bpercent\b",
    ],
    "QUANTITY": [
        r"\bhow many\b",
        r"\btotal number\b",
        r"\bpages scanned\b",
        r"\bnumber of participants\b",
        r"\byears of experience\b",
        r"\bnet pounds\b",
        r"\bpound infeed\b",
        r"\bpounds out\b",
        r"\binfeed\b",
        r"\bout\b",
        r"\bvolume\b",
        r"\blocations\b",
        r"\bcartons\b",
        r"\bstores\b",
        r"\bparticipants lost to follow-up\b",
        r"\bacceptable daily intake\b",
        r"\bresidue limit\b",
        r"\bage of\b",
        r"\by-axis\b",
    ],
    "IDENTIFIER": [
        r"\baccount number\b",
        r"\baccount no\b",
        r"\border number\b",
        r"\bprocedure note number\b",
        r"\bchain id\b",
        r"\bchain id no\b",
        r"\bvenue code\b",
        r"\bbox number\b",
        r"\bqqq number\b",
        r"\bcode\b",
        r"\bid\b",
        r"^\s*what is the [A-Z]{2,6}\??\s*$",
    ],
    "CATEGORY_LABEL": [
        r"\bwhat type of\b",
        r"\bwhat kind of\b",
        r"\btype of research\b",
        r"\bservice provided\b",
        r"\bstand for\b",
        r"\bfull form\b",
        r"\bwhat does .* stand for\b",
        r"\bpesticide used for\b",
        r"\btype of mailing\b",
        r"\bconvenience store\b",
    ],
}

NON_MONEY_NUMERIC_PATTERNS = [
    r"\bnet pounds\b",
    r"\bpound infeed\b",
    r"\bpounds out\b",
    r"\binfeed\b",
    r"\bout\b",
    r"\bparticipants\b",
    r"\bpages scanned\b",
    r"\byears of experience\b",
    r"\bvolume\b",
    r"\blocations\b",
    r"\bcartons\b",
    r"\bstores\b",
    r"\bacceptable daily intake\b",
    r"\bresidue limit\b",
    r"\bage of\b",
    r"\by-axis\b",
]

NON_MONEY_MEASURE_PATTERNS = [
    r"\binterest rate\b",
    r"\bcomposite value\b",
    r"\bproduction\b",
    r"\bpopulation\b",
    r"\bbed days\b",
    r"\bdissolved solids\b",
    r"\bvitamin\b",
    r"\bserum\b",
    r"\bfrequency\b",
    r"\byield\b",
    r"\bpurity\b",
    r"\bvalue on the y axis\b",
    r"\bvalue on the x axis\b",
    r"\bhighest value on the y axis\b",
    r"\bhighest value on the x axis\b",
    r"\blowest value on the y axis\b",
    r"\blowest value on the x axis\b",
    r"\bthickening of conjunctivae\b",
    r"\bself care bed days\b",
]

PERSON_BLOCKLIST_PATTERNS = [
    r"\bmembers?\b",
    r"\bunion\b",
    r"\boffice\b",
    r"\bdepartment\b",
    r"\bcenter\b",
    r"\bcommittee\b",
    r"\bassociation\b",
    r"\bservices\b",
    r"\bprogram\b",
    r"\bstate of\b",
    r"\buniversity\b",
    r"\binstitute\b",
    r"\brestaurant\b",
    r"\bresort\b",
    r"\binn\b",
    r"\bcompany\b",
    r"\bfoundation\b",
    r"\bcalifornia\b",
    r"\bohio\b",
    r"\bmichigan\b",
    r"\bcolorado\b",
    r"\bvirginia\b",
]


# %%
# =========================
# Utils
# =========================

def answer_matches_any_pattern(text: str, patterns: list[str]) -> bool:
    text = normalize_spaces(text).lower()
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def question_is_quantity(q: str) -> bool:
    return contains_any_pattern(q, QUESTION_PATTERNS.get("QUANTITY", []))


def question_is_identifier_like(q: str) -> bool:
    return contains_any_pattern(q, QUESTION_PATTERNS.get("IDENTIFIER", [])) or any(
        kw in q for kw in [
            "pageid",
            "auth. no",
            "auth no",
            "invoice no",
            "voucher no",
            "promo no",
            "item#",
            "item no",
            "upc",
        ]
    )


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def select_answer(example: dict) -> str:
    answers = example.get("answers") or []
    for answer in answers:
        text = str(answer).strip()
        if text:
            return text
    return ""


def safe_text(x: object) -> str:
    return str(x).strip() if x is not None else ""


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", safe_text(text)).strip()


def normalize_label(raw_value: str) -> str:
    value = safe_text(raw_value)
    value = re.sub(r"[`\"'.,:;!?]+", "", value).strip().upper()
    if value in ALLOWED_TYPES:
        return value

    alias_map = {
        "DATE": "DATE_TIME",
        "TIME": "DATE_TIME",
        "YEAR": "DATE_TIME",
        "AMOUNT": "MONEY",
        "NUMBER": "QUANTITY",
        "NUMERIC": "QUANTITY",
        "PHONE": "CONTACT",
        "EMAIL": "CONTACT",
        "FAX": "CONTACT",
        "PERSON": "PERSON_NAME",
        "NAME": "PERSON_NAME",
        "ORG": "ORG_NAME",
        "ORGANIZATION": "ORG_NAME",
        "PAGE_NUMBER": "DOCUMENT_REFERENCE",
        "REFERENCE": "DOCUMENT_REFERENCE",
        "HEADER": "TITLE_HEADER",
        "TITLE": "TITLE_HEADER",
        "ROLE": "ROLE_TITLE",
        "CATEGORY": "CATEGORY_LABEL",
        "TEXT": "FREE_TEXT",
        "SERVICE_TYPE": "CATEGORY_LABEL",
        "ABBREVIATION": "CATEGORY_LABEL",
    }
    return alias_map.get(value, "OTHER")


def contains_any_pattern(text: str, patterns: list[str]) -> bool:
    text = text.lower()
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def is_likely_free_text(answer: str) -> bool:
    a = normalize_spaces(answer)
    if len(a.split()) >= 4:
        if (
            not MONEY_RE.match(a)
            and not PERCENT_RE.match(a)
            and not DATE_LIKE_RE.match(a)
        ):
            return True
    return False


def make_example_id(
    dataset_name: str,
    split: str,
    local_row_id: int,
    question: str,
    answer: str,
) -> str:
    payload = f"{dataset_name}|||{split}|||{local_row_id}|||{normalize_spaces(question)}|||{normalize_spaces(answer)}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def load_existing_results(csv_path: Path) -> pd.DataFrame:
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def save_results(results: list[dict], csv_path: Path, jsonl_path: Path) -> None:
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False, encoding="utf-8")

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_run_paths(split: str) -> dict[str, Path]:
    run_name = FINAL_EXPORT_BASENAMES.get(split, f"{DATASET_NAME.replace('/', '__')}__{split}")

    run_dir = RUNS_DIR / split
    run_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_name": Path(run_name),
        "run_dir": run_dir,
        "output_csv": run_dir / f"{run_name}.csv",
        "output_jsonl": run_dir / f"{run_name}.jsonl",
        "stats_csv": run_dir / f"{run_name}__stats.csv",
        "other_csv": run_dir / f"{run_name}__other_examples.csv",
        "audit_csv": run_dir / f"{run_name}__audit_cases.csv",
        "audit_sample_csv": run_dir / f"{run_name}__audit_sample.csv",
    }


# %%
# =========================
# Rule-based classifier
# =========================

@dataclass
class RuleResult:
    label: Optional[str]
    reason: Optional[str]
    confidence: float


def rule_based_label(question: str, answer: str) -> RuleResult:
    q = normalize_spaces(question).lower()
    a = normalize_spaces(answer)
    a_lower = a.lower()

    if not a:
        return RuleResult("OTHER", "empty_answer", 1.0)

    if a_lower in BOOLEAN_VALUES:
        return RuleResult("BOOLEAN", "boolean_value", 1.0)

    if EMAIL_RE.match(a) or URL_RE.match(a):
        return RuleResult("CONTACT", "email_or_url_regex", 1.0)

    if PERCENT_RE.match(a):
        return RuleResult("PERCENTAGE", "percentage_regex", 1.0)

    # short generic categorical answers
    if a_lower in {"none", "other", "yes", "no", "suspension", "syrup"}:
        return RuleResult("CATEGORY_LABEL", "generic_short_label", 0.9)

    # quantity questions should be resolved before generic date parsing
    if contains_any_pattern(q, QUESTION_PATTERNS.get("QUANTITY", [])) and NUMERIC_RE.match(a):
        return RuleResult("QUANTITY", "question_quantity_numeric", 0.97)

    # explicit document references
    if contains_any_pattern(q, QUESTION_PATTERNS.get("DOCUMENT_REFERENCE", [])):
        if NUMERIC_RE.match(a) or IDENTIFIER_RE.match(a) or re.search(r"\bpage\b", a_lower):
            return RuleResult("DOCUMENT_REFERENCE", "question_doc_reference", 0.98)

    # identifier-like questions before contact/date fallback
    if re.search(r"\b(upc|promo no|item#|item no|code|id|number)\b", q, flags=re.IGNORECASE):
        compact = a.replace(",", "").strip()
        if IDENTIFIER_RE.match(compact):
            return RuleResult("IDENTIFIER", "question_identifier_like", 0.95)

    if contains_any_pattern(q, QUESTION_PATTERNS.get("CONTACT", [])):
        digit_count = len(re.sub(r"\D", "", a))
        if PHONE_RE.match(a) or EMAIL_RE.match(a) or URL_RE.match(a):
            return RuleResult("CONTACT", "question_contact", 0.98)
        if digit_count >= 7:
            return RuleResult("CONTACT", "question_contact_numeric", 0.96)

    if contains_any_pattern(q, QUESTION_PATTERNS.get("DATE_TIME", [])) and DATE_LIKE_RE.match(a):
        return RuleResult("DATE_TIME", "question_date_and_date_like_answer", 0.98)

    if re.search(r"\bexpiration date\b", q, flags=re.IGNORECASE) and DATE_LIKE_RE.match(a):
        return RuleResult("DATE_TIME", "question_expiration_date", 0.98)

    if re.search(r"\bsources?\b", q, flags=re.IGNORECASE):
        return RuleResult("FREE_TEXT", "question_sources", 0.94)

    if re.search(r"\bstand for\b", q, flags=re.IGNORECASE):
        return RuleResult("CATEGORY_LABEL", "abbreviation_expansion", 0.95)

    if re.fullmatch(r"\s*what is [a-z]{2,6}\??\s*", question.strip(), flags=re.IGNORECASE):
        if len(a.split()) <= 4 and not re.search(r"\d", a):
            return RuleResult("CATEGORY_LABEL", "short_abbreviation_like_question", 0.9)

    if contains_any_pattern(q, NON_MONEY_NUMERIC_PATTERNS) and NUMERIC_RE.match(a):
        return RuleResult("QUANTITY", "question_non_money_numeric", 0.97)

    if contains_any_pattern(q, NON_MONEY_MEASURE_PATTERNS) and NUMERIC_RE.match(a):
        return RuleResult("QUANTITY", "question_measure_numeric", 0.96)

    if contains_any_pattern(q, QUESTION_PATTERNS.get("PERCENTAGE", [])) and (PERCENT_RE.match(a) or NUMERIC_RE.match(a)):
        return RuleResult("PERCENTAGE", "question_percentage", 0.95)

    # location before person
    if contains_any_pattern(q, QUESTION_PATTERNS.get("LOCATION", [])):
        if len(a.split()) <= 8 and not re.search(r"\d", a):
            return RuleResult("LOCATION", "question_location", 0.94)

    # explicit time/date spans before identifier fallback
    if re.search(r"\b(time|timing|scheduled|schedule time|working hours|date|when)\b", q, flags=re.IGNORECASE):
        if re.search(r"\d", a) and (
            re.search(r"\b(am|pm|a\.m\.|p\.m\.)\b", a_lower)
            or re.search(r"\d{1,2}[:\-]\d{1,2}", a)
            or re.search(r"[A-Za-z]{3,9}\s+\d{1,2}", a)
            or re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", a)
        ):
            return RuleResult("DATE_TIME", "question_time_or_date_span", 0.97)

    # generic date-like answers, but not for quantity or identifier-like questions
    if DATE_LIKE_RE.match(a):
        if not question_is_quantity(q) and not question_is_identifier_like(q):
            return RuleResult("DATE_TIME", "date_like_regex", 0.92)

    if MONEY_RE.match(a):
        return RuleResult("MONEY", "money_regex", 1.0)

    if contains_any_pattern(q, QUESTION_PATTERNS.get("MONEY", [])) and NUMERIC_RE.match(a):
        return RuleResult("MONEY", "question_money_numeric_answer", 0.96)

    if contains_any_pattern(q, QUESTION_PATTERNS.get("IDENTIFIER", [])):
        if NUMERIC_RE.match(a) or IDENTIFIER_RE.match(a):
            return RuleResult("IDENTIFIER", "question_identifier", 0.95)

    if contains_any_pattern(q, QUESTION_PATTERNS.get("ADDRESS", [])):
        if ZIP_RE.match(a):
            return RuleResult("ADDRESS", "question_address_zip", 0.92)
        return RuleResult("ADDRESS", "question_address", 0.9)

    if contains_any_pattern(q, QUESTION_PATTERNS.get("ORG_NAME", [])):
        if len(a.split()) <= 12 and not DATE_LIKE_RE.match(a):
            return RuleResult("ORG_NAME", "question_org_name", 0.94)

    if contains_any_pattern(q, QUESTION_PATTERNS.get("PERSON_NAME", [])):
        if (
            not re.search(r"\d", a)
            and len(a.split()) <= 8
            and not answer_matches_any_pattern(a, PERSON_BLOCKLIST_PATTERNS)
        ):
            return RuleResult("PERSON_NAME", "question_person_name", 0.94)

    if contains_any_pattern(q, QUESTION_PATTERNS.get("ROLE_TITLE", [])):
        return RuleResult("ROLE_TITLE", "question_role_title", 0.94)

    if contains_any_pattern(q, QUESTION_PATTERNS.get("TITLE_HEADER", [])):
        if len(a.split()) <= 20:
            if re.search(r"\bfootnote\b", q, flags=re.IGNORECASE):
                return RuleResult("FREE_TEXT", "question_footnote_text", 0.93)
            return RuleResult("TITLE_HEADER", "question_title_header", 0.93)

    if contains_any_pattern(q, QUESTION_PATTERNS.get("CATEGORY_LABEL", [])):
        if len(a.split()) <= 10 and not re.search(r"\d", a):
            return RuleResult("CATEGORY_LABEL", "question_category_label", 0.92)

    if PHONE_RE.match(a) and len(re.sub(r"\D", "", a)) >= 7:
        return RuleResult("CONTACT", "phone_regex", 0.95)

    compact = a.replace(",", "").strip()
    if IDENTIFIER_RE.match(compact):
        has_letters = bool(re.search(r"[A-Za-z]", compact))
        has_digits = bool(re.search(r"\d", compact))
        if has_letters and has_digits:
            if re.search(r"\bdate\b", q, flags=re.IGNORECASE) and DATE_LIKE_RE.match(a):
                return RuleResult("DATE_TIME", "question_date_over_identifier", 0.95)
            return RuleResult("IDENTIFIER", "alphanumeric_identifier_regex", 0.9)

    if NUMERIC_RE.match(a):
        return RuleResult("QUANTITY", "plain_numeric", 0.85)

    # short nominal answers that are not numbers
    if len(a.split()) <= 4 and not re.search(r"\d", a):
        return RuleResult("CATEGORY_LABEL", "short_text_category_like", 0.82)

    if is_likely_free_text(a):
        return RuleResult("FREE_TEXT", "long_text_span", 0.8)

    return RuleResult(None, None, 0.0)


# %%
# =========================
# GigaChat client
# =========================

load_env_file(ENV_PATH)

GIGACHAT_CREDENTIALS = os.environ.get("GIGACHAT_CREDENTIALS", "").strip()
GIGACHAT_SCOPE = os.environ.get("GIGACHAT_SCOPE", "GIGACHAT_API_PERS").strip()

if not GIGACHAT_CREDENTIALS:
    raise RuntimeError("Environment variable GIGACHAT_CREDENTIALS is required.")

client = GigaChat(
    credentials=GIGACHAT_CREDENTIALS,
    scope=GIGACHAT_SCOPE,
    model=MODEL,
    verify_ssl_certs=VERIFY_SSL_CERTS,
)


def classify_with_llm(question: str, answer: str, max_retries: int = 5) -> tuple[str, str]:
    prompt = PROMPT_TEMPLATE.format(
        question=normalize_spaces(question),
        answer=normalize_spaces(answer),
    )

    payload = Chat(
        messages=[Messages(role=MessagesRole.USER, content=prompt)],
        temperature=0,
    )

    last_err = None
    for attempt in range(max_retries):
        try:
            response = client.chat(payload)
            raw_label = safe_text(response.choices[0].message.content)
            norm_label = normalize_label(raw_label)
            return norm_label, raw_label
        except Exception as e:
            last_err = e
            sleep_t = min(2 ** attempt, 20) + random.random() * 0.5
            time.sleep(sleep_t)

    raise RuntimeError(f"LLM labeling failed after {max_retries} retries: {last_err}")


# %%
# =========================
# Dataset loading
# =========================

def prepare_records(
    dataset_name: str,
    split: str,
    limit: Optional[int],
) -> list[dict]:
    dataset = load_dataset(dataset_name, split=split)

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    records = []
    for idx, example in enumerate(dataset):
        question = safe_text(example.get("question"))
        answer = select_answer(example)

        example_id = make_example_id(
            dataset_name=dataset_name,
            split=split,
            local_row_id=idx,
            question=question,
            answer=answer,
        )

        records.append(
            {
                "dataset_name": dataset_name,
                "split": split,
                "local_row_id": idx,
                "example_id": example_id,
                "question": question,
                "answer": answer,
            }
        )

    return records


# %%
# =========================
# Main run per split
# =========================

def run_labeling_for_split(
    split: str,
    limit: Optional[int] = None,
    resume: bool = True,
) -> pd.DataFrame:
    print("=" * 80)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Split: {split}")
    print(f"Limit: {limit}")
    print(f"Resume: {resume}")

    paths = build_run_paths(split)
    output_csv = paths["output_csv"]
    output_jsonl = paths["output_jsonl"]
    stats_csv = paths["stats_csv"]
    other_csv = paths["other_csv"]
    audit_csv = paths["audit_csv"]
    audit_sample_csv = paths["audit_sample_csv"]

    records = prepare_records(DATASET_NAME, split, limit)
    print(f"Prepared records: {len(records)}")

    results: list[dict] = []
    done_ids: set[str] = set()

    if resume and output_csv.exists():
        existing_df = load_existing_results(output_csv)
        if len(existing_df) > 0 and "example_id" in existing_df.columns:
            existing_df["example_id"] = existing_df["example_id"].astype(str)
            done_ids = set(existing_df["example_id"].tolist())
            results = existing_df.to_dict(orient="records")
            print(f"Loaded existing rows: {len(results)}")
        else:
            print("Existing CSV found, but missing required columns. Starting fresh.")

    pending_records = [r for r in records if r["example_id"] not in done_ids]
    print(f"Pending rows: {len(pending_records)}")

    random.seed(RANDOM_SEED)

    for n, row in enumerate(tqdm(pending_records, desc=f"Labeling {split}"), start=1):
        question = row["question"]
        answer = row["answer"]

        field_type = "OTHER"
        raw_llm_output = ""
        source = ""
        rule_reason = ""
        rule_confidence = 0.0
        error_message = ""

        try:
            rr = rule_based_label(question, answer)

            if USE_RULES_FIRST and rr.label is not None and rr.confidence >= RULE_ACCEPT_THRESHOLD:
                field_type = rr.label
                source = "rule"
                rule_reason = rr.reason or ""
                rule_confidence = rr.confidence
            else:
                field_type, raw_llm_output = classify_with_llm(question, answer)
                source = "llm"
                rule_reason = rr.reason or ""
                rule_confidence = rr.confidence

        except Exception as e:
            field_type = "OTHER"
            source = "error"
            error_message = str(e)

        result_row = {
            "dataset_name": row["dataset_name"],
            "split": row["split"],
            "local_row_id": row["local_row_id"],
            "example_id": row["example_id"],
            "question": question,
            "answer": answer,
            "field_type": field_type,
            "field_group": GROUP_MAP.get(field_type, "OPEN_TEXT"),
            "sensitivity": SENSITIVITY_MAP.get(field_type, "LOW"),
            "label_source": source,
            "rule_reason": rule_reason,
            "rule_confidence": rule_confidence,
            "raw_llm_output": raw_llm_output if STORE_RAW_LLM_OUTPUT else "",
            "error_message": error_message,
        }

        results.append(result_row)

        if n % SAVE_EVERY == 0:
            save_results(results, output_csv, output_jsonl)

        time.sleep(SLEEP_SEC)

    save_results(results, output_csv, output_jsonl)
    print(f"Saved {len(results)} rows to: {output_csv}")

    results_df = pd.DataFrame(results)

    field_stats = (
        results_df["field_type"]
        .value_counts(dropna=False)
        .rename_axis("field_type")
        .reset_index(name="count")
    )

    source_stats = (
        results_df["label_source"]
        .value_counts(dropna=False)
        .rename_axis("label_source")
        .reset_index(name="count")
    )

    group_stats = (
        results_df["field_group"]
        .value_counts(dropna=False)
        .rename_axis("field_group")
        .reset_index(name="count")
    )

    sensitivity_stats = (
        results_df["sensitivity"]
        .value_counts(dropna=False)
        .rename_axis("sensitivity")
        .reset_index(name="count")
    )

    stats_df = pd.concat(
        [
            field_stats.assign(stat_type="field_type", stat_value=field_stats["field_type"])[["stat_type", "stat_value", "count"]],
            source_stats.assign(stat_type="label_source", stat_value=source_stats["label_source"])[["stat_type", "stat_value", "count"]],
            group_stats.assign(stat_type="field_group", stat_value=group_stats["field_group"])[["stat_type", "stat_value", "count"]],
            sensitivity_stats.assign(stat_type="sensitivity", stat_value=sensitivity_stats["sensitivity"])[["stat_type", "stat_value", "count"]],
        ],
        ignore_index=True,
    )

    stats_df.to_csv(stats_csv, index=False, encoding="utf-8")

    print("\nField type statistics:")
    display(field_stats)

    print("\nLabel source statistics:")
    display(source_stats)

    print("\nField group statistics:")
    display(group_stats)

    print("\nSensitivity statistics:")
    display(sensitivity_stats)

    other_df = results_df.loc[
        results_df["field_type"] == "OTHER",
        [
            "example_id",
            "split",
            "local_row_id",
            "question",
            "answer",
            "field_type",
            "label_source",
            "raw_llm_output",
            "error_message",
        ],
    ].reset_index(drop=True)

    print(f"\nOTHER examples: {len(other_df)}")
    display(other_df.head(50))
    other_df.to_csv(other_csv, index=False, encoding="utf-8")

    audit_df = results_df.loc[
        (results_df["label_source"] == "llm") & (results_df["rule_confidence"] > 0),
        [
            "example_id",
            "split",
            "local_row_id",
            "question",
            "answer",
            "field_type",
            "rule_reason",
            "rule_confidence",
            "raw_llm_output",
        ],
    ].sort_values(by="rule_confidence", ascending=False).reset_index(drop=True)

    print(f"\nLLM-overridden or LLM-needed cases with nonzero rule confidence: {len(audit_df)}")
    display(audit_df.head(50))
    audit_df.to_csv(audit_csv, index=False, encoding="utf-8")

    if len(results_df) >= 200:
        audit_sample = results_df.sample(200, random_state=42)
    else:
        audit_sample = results_df.copy()
    audit_sample.to_csv(audit_sample_csv, index=False, encoding="utf-8")

    return results_df


# %%
# =========================
# Merge all splits
# =========================

def merge_all_split_outputs(splits: list[str]) -> pd.DataFrame:
    all_dfs = []

    for split in splits:
        paths = build_run_paths(split)
        csv_path = paths["output_csv"]
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            all_dfs.append(df)

    if not all_dfs:
        print("No split outputs found to merge.")
        return pd.DataFrame()

    merged_df = pd.concat(all_dfs, ignore_index=True)

    merged_csv = MERGED_DIR / f"{DATASET_NAME.replace('/', '__')}__all_splits.csv"
    merged_jsonl = MERGED_DIR / f"{DATASET_NAME.replace('/', '__')}__all_splits.jsonl"

    merged_df.to_csv(merged_csv, index=False, encoding="utf-8")
    with merged_jsonl.open("w", encoding="utf-8") as f:
        for row in merged_df.to_dict(orient="records"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Merged rows: {len(merged_df)}")
    print(f"Merged CSV: {merged_csv}")
    print(f"Merged JSONL: {merged_jsonl}")

    return merged_df


# %%
# =========================
# Golden evaluation
# =========================

def resolve_gold_path(split: str) -> Path:
    candidates = [
        GOLD_DIR / f"docvqa_gold_labels_{split}_v1.csv",
        GOLD_DIR / "docvqa_gold_labels_v1.csv",
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Gold-файл не найден для split={split}. Проверены пути: {candidates}"
    )


def run_gold_evaluation_for_split(split: str) -> None:
    paths = build_run_paths(split)
    pred_path = paths["output_csv"]

    if not pred_path.exists():
        raise FileNotFoundError(f"Не найден файл предсказаний для split={split}: {pred_path}")

    pred_df = pd.read_csv(pred_path)
    gold_path = resolve_gold_path(split)
    gold_df = pd.read_csv(gold_path)

    print(f"\nЗагружен prediction-файл: {pred_path}")
    print(f"Количество строк в prediction-файле: {len(pred_df)}")

    print(f"\nЗагружен gold-файл: {gold_path}")
    print(f"Количество строк в gold-файле: {len(gold_df)}")
    display(gold_df.head(10))

    pred_has_example_id = "example_id" in pred_df.columns
    gold_has_example_id = "example_id" in gold_df.columns

    if gold_has_example_id and pred_has_example_id:
        eval_df = gold_df.merge(
            pred_df,
            on="example_id",
            how="left",
            suffixes=("_goldfile", ""),
        )
    else:
        if "local_row_id" in gold_df.columns and "local_row_id" in pred_df.columns:
            eval_df = gold_df.merge(
                pred_df,
                on=["local_row_id"],
                how="left",
                suffixes=("_goldfile", ""),
            )
        elif "row_id" in gold_df.columns and "local_row_id" in pred_df.columns:
            gold_df = gold_df.rename(columns={"row_id": "local_row_id"})
            eval_df = gold_df.merge(
                pred_df,
                on=["local_row_id"],
                how="left",
                suffixes=("_goldfile", ""),
            )
        else:
            raise RuntimeError(
                "Не удалось сматчить gold и predictions. "
                "Добавь example_id в gold или хотя бы local_row_id/row_id."
            )

    if "gold_field_type" not in eval_df.columns:
        if "field_type_goldfile" in eval_df.columns:
            eval_df["gold_field_type"] = eval_df["field_type_goldfile"]
        else:
            raise RuntimeError("В gold-файле не найден gold_field_type.")

    for col in ["gold_field_type", "field_type", "question", "answer"]:
        if col in eval_df.columns:
            eval_df[col] = eval_df[col].astype(str).str.strip()

    eval_df = eval_df[
        eval_df["gold_field_type"].notna()
        & (eval_df["gold_field_type"] != "")
        & eval_df["field_type"].notna()
        & (eval_df["field_type"] != "")
    ].reset_index(drop=True)

    print(f"\nКоличество примеров для оценки: {len(eval_df)}")
    display(eval_df.head(10))

    acc = accuracy_score(eval_df["gold_field_type"], eval_df["field_type"])
    print(f"Accuracy: {acc:.4f}")

    report_text = classification_report(
        eval_df["gold_field_type"],
        eval_df["field_type"],
        digits=4,
        zero_division=0,
    )
    print("\nClassification report:")
    print(report_text)

    labels = sorted(set(eval_df["gold_field_type"]) | set(eval_df["field_type"]))
    cm = confusion_matrix(
        eval_df["gold_field_type"],
        eval_df["field_type"],
        labels=labels,
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"gold::{label}" for label in labels],
        columns=[f"pred::{label}" for label in labels],
    )

    print("Confusion matrix:")
    display(cm_df)

    cols_for_errors = [
        c for c in [
            "example_id",
            "split",
            "local_row_id",
            "question",
            "answer",
            "field_type",
            "gold_field_type",
            "label_source",
            "rule_reason",
            "rule_confidence",
            "annotator_notes",
        ] if c in eval_df.columns
    ]

    errors_df = eval_df.loc[
        eval_df["gold_field_type"] != eval_df["field_type"],
        cols_for_errors,
    ].reset_index(drop=True)

    print(f"Количество ошибок: {len(errors_df)}")
    display(errors_df.head(50))

    for source in ["rule", "llm", "error"]:
        if "label_source" not in eval_df.columns:
            continue
        sub = eval_df[eval_df["label_source"] == source].copy()
        if len(sub) == 0:
            continue

        sub_acc = accuracy_score(sub["gold_field_type"], sub["field_type"])
        print(f"{source}: n={len(sub)}, accuracy={sub_acc:.4f}")

    split_prefix = f"{DATASET_NAME.replace('/', '__')}__{split}"

    errors_csv = EVAL_DIR / f"{split_prefix}__gold_errors.csv"
    cm_csv = EVAL_DIR / f"{split_prefix}__gold_confusion_matrix.csv"
    metrics_txt = EVAL_DIR / f"{split_prefix}__gold_metrics.txt"
    eval_merged_csv = EVAL_DIR / f"{split_prefix}__gold_eval_merged.csv"

    errors_df.to_csv(errors_csv, index=False, encoding="utf-8")
    cm_df.to_csv(cm_csv, encoding="utf-8")
    eval_df.to_csv(eval_merged_csv, index=False, encoding="utf-8")

    with metrics_txt.open("w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.6f}\n\n")
        f.write("Classification report:\n")
        f.write(report_text)

    print(f"Сохранены ошибки: {errors_csv}")
    print(f"Сохранена confusion matrix: {cm_csv}")
    print(f"Сохранены метрики: {metrics_txt}")
    print(f"Сохранён merged eval: {eval_merged_csv}")

    if len(errors_df) > 0:
        pair_errors = (
            errors_df.groupby(["gold_field_type", "field_type"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .reset_index(drop=True)
        )

        pair_errors_csv = EVAL_DIR / f"{split_prefix}__gold_error_pairs.csv"
        pair_errors.to_csv(pair_errors_csv, index=False, encoding="utf-8")

        print("Самые частые пары ошибок:")
        display(pair_errors.head(20))
        print(f"Сохранены пары ошибок: {pair_errors_csv}")
    else:
        print("Ошибок нет.")


# %%
# =========================
# Main execution
# =========================

all_split_results = []

for split in SPLITS:
    limit = LIMITS.get(split, None)
    df_split = run_labeling_for_split(
        split=split,
        limit=limit,
        resume=RESUME,
    )
    all_split_results.append(df_split)

merged_df = merge_all_split_outputs(SPLITS)

if RUN_GOLD_EVAL:
    for split in GOLD_EVAL_SPLITS:
        try:
            run_gold_evaluation_for_split(split)
        except Exception as e:
            print(f"[WARN] Gold evaluation failed for split={split}: {e}")

    error_tables = []
    for split in GOLD_EVAL_SPLITS:
        split_prefix = f"{DATASET_NAME.replace('/', '__')}__{split}"
        errors_csv = EVAL_DIR / f"{split_prefix}__gold_errors.csv"
        if not errors_csv.exists():
            continue

        split_errors_df = pd.read_csv(errors_csv)
        if "split" not in split_errors_df.columns:
            split_errors_df.insert(0, "split", split)
        error_tables.append(split_errors_df)

    print("\nИтоговая таблица ошибок:")
    if error_tables:
        final_errors_df = pd.concat(error_tables, ignore_index=True)
        display(final_errors_df)
    else:
        print("Файлы с ошибками не найдены.")