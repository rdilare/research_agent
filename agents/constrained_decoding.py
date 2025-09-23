"""
JSON-based constrained decoding (single-pass, no retries).

This module guides the LLM to emit JSON-only output via prompt instructions and
then strictly parses and validates it into Pydantic models. No retry loops are
used; invalid outputs raise JSONDecodingError and the caller may handle fallbacks.

Public API (consumed by graph.py):
 - class JSONConstrainedParser
 - class JSONConstrainedLLMChain
 - class JSONDecodingError(Exception)
 - def create_json_parser(pydantic_model, max_retries, allow_partial, fill_defaults)
 - def create_json_chain(llm, pydantic_model, max_retries, allow_partial, fill_defaults)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Type
import json
import re

try:
	# Pydantic v2
	from pydantic import BaseModel
	V2 = True
except Exception:  # pragma: no cover
	from pydantic.v1 import BaseModel  # type: ignore
	V2 = False


class JSONDecodingError(Exception):
	"""Raised when structured JSON decoding fails (single-pass, no retries)."""


def _model_json_schema(model_cls: Type[BaseModel]) -> Dict[str, Any]:
	if V2:
		return model_cls.model_json_schema()
	return model_cls.schema()  # type: ignore[attr-defined]


def _model_validate_from_obj(model_cls: Type[BaseModel], obj: Any) -> BaseModel:
	if V2:
		return model_cls.model_validate(obj)
	return model_cls.parse_obj(obj)  # type: ignore[attr-defined]


def _extract_json_payload(text: str) -> Optional[str]:
	"""Extract a plausible JSON object/array from text without retries.

	- If fenced code blocks are present, extract the fenced JSON.
	- Else, scan for the first balanced object {...} or array [...].
	Returns the JSON substring or None if not found.
	"""
	if not text:
		return None

	# Fenced code block (```json ... ``` or ``` ... ```)
	m = re.search(r"```(?:json)?\s*([\[{][\s\S]*[\]}])\s*```", text, flags=re.IGNORECASE)
	if m:
		return m.group(1)

	# First balanced object
	start = text.find("{")
	if start != -1:
		depth = 0
		for i in range(start, len(text)):
			c = text[i]
			if c == '{':
				depth += 1
			elif c == '}':
				depth -= 1
				if depth == 0:
					return text[start:i+1]

	# Or first balanced array
	start = text.find("[")
	if start != -1:
		depth = 0
		for i in range(start, len(text)):
			c = text[i]
			if c == '[':
				depth += 1
			elif c == ']':
				depth -= 1
				if depth == 0:
					return text[start:i+1]
	return None


class JSONConstrainedParser:
	"""Generates format instructions and validates JSON into a Pydantic model."""

	def __init__(
		self,
		model_cls: Type[BaseModel],
		allow_partial: bool = True,
		fill_defaults: bool = True,
	) -> None:
		self.model_cls = model_cls
		self.allow_partial = allow_partial
		self.fill_defaults = fill_defaults
		self._schema = _model_json_schema(model_cls)

	def get_format_instructions(self) -> str:
		schema_str = json.dumps(self._schema, indent=2)
		return (
			"You MUST respond with a single valid JSON object that conforms to the "
			"following JSON Schema. Do not include any prose before or after the JSON.\n\n"
			f"JSON Schema:\n{schema_str}\n"
		)

	def parse(self, text: str) -> BaseModel:
		# One-pass parsing: try to extract JSON from the output; if not found, use raw text
		payload = _extract_json_payload(text) or text
		obj = json.loads(payload)
		return _model_validate_from_obj(self.model_cls, obj)


class JSONConstrainedLLMChain:
	"""Run an LLM once and validate the JSON output against the Pydantic model."""

	def __init__(
		self,
		llm: Any,
		parser: JSONConstrainedParser,
		max_retries: int = 3,  # kept for API parity; unused
	) -> None:
		self.llm = llm
		self.parser = parser

	def run(self, prompt: str) -> BaseModel:
		# Single-pass call, no retries. Prepend a short guard for JSON-only output.
		final_prompt = (
			"Return ONLY a valid JSON object. Do not include any text before or after the JSON.\n\n"
			+ prompt
		)

		# LangChain LLM compatibility
		try:
			if hasattr(self.llm, "invoke"):
				raw = self.llm.invoke(final_prompt)
			elif hasattr(self.llm, "predict"):
				raw = self.llm.predict(final_prompt)
			else:
				raw = self.llm(final_prompt)  # type: ignore[call-arg]
		except Exception as e:
			raise JSONDecodingError(f"LLM invocation failed: {e}") from e

		# Normalize raw to text
		if hasattr(raw, "content"):
			text = raw.content  # e.g., AIMessage
		else:
			text = str(raw)

		# Strict one-pass parsing and validation
		try:
			return self.parser.parse(text)
		except Exception as e:
			raise JSONDecodingError(f"Invalid JSON structured output: {e}") from e


def create_json_parser(
	pydantic_model: Type[BaseModel],
	max_retries: int = 3,  # signature parity only
	allow_partial: bool = True,
	fill_defaults: bool = True,
) -> JSONConstrainedParser:
	return JSONConstrainedParser(
		model_cls=pydantic_model,
		allow_partial=allow_partial,
		fill_defaults=fill_defaults,
	)


def create_json_chain(
	llm: Any,
	pydantic_model: Type[BaseModel],
	max_retries: int = 3,  # signature parity only
	allow_partial: bool = True,
	fill_defaults: bool = True,
) -> JSONConstrainedLLMChain:
	parser = create_json_parser(
		pydantic_model=pydantic_model,
		max_retries=max_retries,
		allow_partial=allow_partial,
		fill_defaults=fill_defaults,
	)
	return JSONConstrainedLLMChain(llm=llm, parser=parser, max_retries=max_retries)

