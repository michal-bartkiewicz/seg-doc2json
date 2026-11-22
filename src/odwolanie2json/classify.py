from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class Classification:
    document_type: str
    score: int
    signals: tuple[str, ...]


class ClassificationError(ValueError):
    pass


@dataclass(frozen=True)
class SignalRule:
    """A weighted regex signal scoped to a document, bounded header, or title line."""

    name: str
    pattern: str
    weight: int
    scope: str = "document"


@dataclass(frozen=True)
class DocumentTypeProfile:
    """Classification rules and acceptance requirements for one document type."""

    signals: tuple[SignalRule, ...]
    minimum_score: int = 5
    minimum_margin: int = 2
    # Each inner tuple is a group of signals that must all match. Any one group
    # is sufficient. This lets a profile require structural evidence as well as
    # a high weighted score, so a passing mention in a long document is not enough.
    required_signal_groups: tuple[tuple[str, ...], ...] = ()
    # A profile is ineligible when every signal in any excluded group matches.
    excluded_signal_groups: tuple[tuple[str, ...], ...] = ()


DEFAULT_CLASSIFICATION_PROFILES: dict[str, DocumentTypeProfile] = {
    "odwolanie": DocumentTypeProfile(
        signals=(
            SignalRule("appeal_title", r"\bodwolani\w*\b", 7, "header"),
            SignalRule("kio_name", r"\bkrajowej\s+izby\s+odwolawczej\b", 4),
            SignalRule("kio_addressee", r"\bprezes\w*\s+krajowej\s+izby\s+odwolawczej\b", 3),
            SignalRule("kio_short_addressee", r"\b(?:do\s+)?prezes\w*\s+kio\b", 5, "header"),
            SignalRule("appealing_party", r"\bodwoluj\w*\b", 2, "header"),
            SignalRule("contracting_authority", r"\bzamawiaj\w*\b", 2, "header"),
            SignalRule(
                "procurement_context",
                r"\b(?:postepowani\w*|bzp|ted|przedmiot)\b",
                2,
                "header",
            ),
            SignalRule("appeal_action", r"\b(?:zaskarzon\w*|czynnos\w*)\b", 1, "header"),
            SignalRule("appeal_claims", r"\bzarzut\w*\b", 1),
            SignalRule(
                "kio_decision_heading",
                r"^\s*(?:wyrok|postanowienie)\b",
                0,
                "opening_headings",
            ),
        ),
        required_signal_groups=(
            ("appealing_party", "contracting_authority", "appeal_title"),
            ("appealing_party", "contracting_authority", "kio_addressee"),
            ("appealing_party", "contracting_authority", "kio_short_addressee"),
            ("appealing_party", "contracting_authority", "appeal_action"),
        ),
        excluded_signal_groups=(("kio_decision_heading",),),
    ),
    "pelnomocnictwo": DocumentTypeProfile(
        signals=(
            SignalRule("power_title", r"\bpelnomocnictwo\b", 8, "opening_headings"),
            SignalRule(
                "grant_clause",
                r"\budziel\w*.{0,160}\bpelnomocnictw\w*\b",
                5,
                "header",
            ),
            SignalRule("grant_action", r"\budziel\w*\b", 3),
            SignalRule("principal", r"\bmocodawc\w*\b", 2),
            SignalRule(
                "attorney",
                r"\b(?:pelnomocnik\w*|adwokat\w*|radca\s+prawny|prokurent\w*|"
                r"reprezentowan\w*|mocobiorc\w*)\b",
                2,
            ),
            SignalRule(
                "other_document_heading",
                r"^\s*(?:(?:wzor|projekt)\s+)?(?:umow\w*|kontrakt\w*|faktur\w*|"
                r"rachunk\w*|wyrok\w*|postanow\w*|aneks\w*|regulamin\w*|protokol\w*)\b",
                0,
                "opening_headings",
            ),
        ),
        required_signal_groups=(
            ("power_title", "attorney"),
            ("power_title", "principal", "grant_action"),
            ("grant_clause", "attorney"),
            ("grant_clause", "principal"),
        ),
        excluded_signal_groups=(("other_document_heading",),),
    ),
    "przelew": DocumentTypeProfile(
        signals=(
            SignalRule(
                "transfer_confirmation",
                r"\bpotwierdzenie\b.{0,80}\bprzelew\w*\b",
                10,
                "header",
            ),
            SignalRule(
                "transaction_confirmation",
                r"\bpotwierdzenie\b.{0,80}\btransakcj\w*\b",
                10,
                "header",
            ),
            SignalRule("transfer_keyword", r"\bprzelew\w*\b", 5, "header"),
            SignalRule(
                "transfer_detail",
                r"\b(?:kwota|rachunek\w*|operacj\w*|transakcj\w*|tytul\w*|platnosci\w*)\b",
                2,
                "header",
            ),
            SignalRule(
                "account_number",
                r"(?<!\d)\d(?:[\s-]*\d){25}(?!\d)",
                3,
            ),
            SignalRule(
                "operation_date",
                r"\bdata\s+(?:i\s+godzina\s+)?operacji\b",
                4,
            ),
            SignalRule("transaction_title", r"\btytul\s+transakcji\b", 3),
            SignalRule("transfer_amount", r"\bkwota\s+(?:transakcji|przelewu)\b", 3),
            SignalRule(
                "card_payment_heading",
                r"^\s*(?:potwierdzenie\s+(?:transakcji|platnosci).{0,80}\b(?:kart\w*|card)\b|"
                r"(?:platnosc|payment)\s+(?:kart\w*|card)\b|terminal\s+pos\b)",
                0,
                "opening_headings",
            ),
        ),
        required_signal_groups=(
            ("transfer_confirmation", "account_number"),
            ("transfer_confirmation", "operation_date"),
            ("transfer_confirmation", "transaction_title"),
            ("transfer_confirmation", "transfer_amount"),
            ("transaction_confirmation", "account_number"),
            ("transfer_keyword", "transfer_detail", "account_number"),
        ),
        excluded_signal_groups=(("card_payment_heading",),),
    ),
}


def _normalize(text: str) -> str:
    # Polish ł/Ł has no NFKD decomposition, so transliterate it explicitly.
    # OCR sometimes inserts a hyphen at a line wrap inside a word.
    text = re.sub(r"(?<=[^\W\d_])-[ \t]*\r?\n[ \t]*(?=[^\W\d_])", "", text)
    text = text.replace("ł", "l").replace("Ł", "L")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", text).casefold()


def _header_text(text: str) -> str:
    """Return a bounded opening block for title and party-role cues."""
    lines: list[str] = []
    for line in text.splitlines():
        if line.strip():
            lines.append(line)
        if len(lines) >= 60:
            break
    return "\n".join(lines)[:2000]


def _title_text(text: str) -> str:
    """Return the first non-empty line for headings that rule out another type."""
    return next((line.strip() for line in text.splitlines() if line.strip()), "")[:300]


def _opening_headings_text(text: str, *, max_lines: int = 16, max_chars: int = 1200) -> str:
    """Keep a few opening lines so OCR preambles cannot hide the actual heading."""
    lines = [line.strip() for line in text.splitlines() if line.strip()][:max_lines]
    return "\n".join(lines)[:max_chars]


def _normalize_title(text: str) -> str:
    return "\n".join(_normalize(line) for line in text.splitlines() if line.strip())


class RegexDocumentClassifier:
    """Configurable weighted-regex classifier with optional structural anchors."""

    def __init__(self, profiles: Mapping[str, DocumentTypeProfile] | None = None) -> None:
        self._profiles: dict[str, DocumentTypeProfile] = {}
        self._compiled: dict[str, tuple[tuple[SignalRule, re.Pattern[str]], ...]] = {}
        for document_type, profile in (profiles or {}).items():
            self.register(document_type, profile)

    @property
    def document_types(self) -> frozenset[str]:
        return frozenset(self._profiles)

    def register(
        self,
        document_type: str,
        profile: DocumentTypeProfile,
        *,
        replace: bool = False,
    ) -> None:
        """Register a profile; duplicate types are rejected unless replace=True."""
        document_type = document_type.strip()
        if not document_type:
            raise ValueError("Document type must not be empty.")
        if document_type in self._profiles and not replace:
            raise ValueError(f"Document type is already registered: {document_type}")
        if profile.minimum_score < 0 or profile.minimum_margin < 0:
            raise ValueError("Classification thresholds must be non-negative.")
        if not profile.signals:
            raise ValueError("A document type profile must define at least one signal.")
        if any(not signal.name.strip() for signal in profile.signals):
            raise ValueError("Signal names must not be empty.")
        if any(not signal.pattern.strip() for signal in profile.signals):
            raise ValueError("Signal patterns must not be empty.")
        names = [signal.name for signal in profile.signals]
        if len(names) != len(set(names)):
            raise ValueError(f"Signal names must be unique for document type: {document_type}")
        if any(
            signal.scope not in {"document", "header", "title", "opening_headings"}
            for signal in profile.signals
        ):
            raise ValueError(
                "Signal scope must be 'document', 'header', 'title' or 'opening_headings'."
            )
        groups = (*profile.required_signal_groups, *profile.excluded_signal_groups)
        unknown = {
            signal_name
            for group in groups
            for signal_name in group
            if signal_name not in names
        }
        if any(not group for group in groups):
            raise ValueError("Required and excluded signal groups must not be empty.")
        if unknown:
            raise ValueError(
                f"Required signal groups reference unknown signals for {document_type}: "
                + ", ".join(sorted(unknown))
            )
        try:
            compiled = tuple(
                (
                    signal,
                    re.compile(
                        signal.pattern,
                        re.IGNORECASE | re.DOTALL | re.MULTILINE | re.UNICODE,
                    ),
                )
                for signal in profile.signals
            )
        except re.error as exc:
            raise ValueError(f"Invalid regex for document type {document_type}.") from exc
        self._profiles[document_type] = profile
        self._compiled[document_type] = compiled

    @staticmethod
    def _has_required_group(profile: DocumentTypeProfile, matched: set[str]) -> bool:
        return not profile.required_signal_groups or any(
            set(group).issubset(matched) for group in profile.required_signal_groups
        )

    @staticmethod
    def _has_excluded_group(profile: DocumentTypeProfile, matched: set[str]) -> bool:
        return any(set(group).issubset(matched) for group in profile.excluded_signal_groups)

    def classify(self, text: str) -> Classification:
        if not text.strip():
            raise ClassificationError("Document has no text to classify.")

        normalized = _normalize(text)
        normalized_header = _normalize(_header_text(text))
        normalized_title = _normalize_title(_title_text(text))
        normalized_opening_headings = "\n".join(
            _normalize(line)
            for line in _opening_headings_text(text).splitlines()
            if line.strip()
        )
        scores: dict[str, int] = {}
        signals_by_type: dict[str, tuple[str, ...]] = {}
        eligible: list[tuple[str, int]] = []
        for document_type, profile in self._profiles.items():
            matched = tuple(
                signal.name
                for signal, pattern in self._compiled[document_type]
                if pattern.search(
                    normalized_header
                    if signal.scope == "header"
                    else normalized_title
                    if signal.scope == "title"
                    else normalized_opening_headings
                    if signal.scope == "opening_headings"
                    else normalized
                )
            )
            score = sum(
                signal.weight
                for signal, _ in self._compiled[document_type]
                if signal.name in matched
            )
            scores[document_type] = score
            signals_by_type[document_type] = matched
            matched_set = set(matched)
            if self._has_required_group(profile, matched_set) and not self._has_excluded_group(
                profile, matched_set
            ):
                eligible.append((document_type, score))

        if not eligible:
            details = ", ".join(f"{kind}={value}" for kind, value in sorted(scores.items()))
            raise ClassificationError(
                "No document type matched its required structural signals"
                + (f" ({details})." if details else ".")
            )

        ranked = sorted(eligible, key=lambda item: item[1], reverse=True)
        winner, score = ranked[0]
        profile = self._profiles[winner]
        runner_up_score = ranked[1][1] if len(ranked) > 1 else 0
        if score < profile.minimum_score or score - runner_up_score < profile.minimum_margin:
            details = ", ".join(f"{kind}={value}" for kind, value in ranked)
            raise ClassificationError(
                f"Document type is ambiguous under local regex rules ({details})."
            )
        return Classification(winner, score, signals_by_type[winner])


DEFAULT_CLASSIFIER = RegexDocumentClassifier(DEFAULT_CLASSIFICATION_PROFILES)


def classify_document_type(text: str) -> Classification:
    """Classify a supported document using weighted signals and header anchors."""
    return DEFAULT_CLASSIFIER.classify(text)
