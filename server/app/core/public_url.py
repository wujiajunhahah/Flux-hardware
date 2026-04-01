from __future__ import annotations

from fastapi import Request
from starlette.datastructures import URL

from .settings import settings


def public_url_for(request: Request, route_name: str, **path_params: str) -> URL:
    return apply_public_base(request, request.url_for(route_name, **path_params))


def apply_public_base(request: Request, url: URL) -> URL:
    if settings.public_base_url:
        base = URL(settings.public_base_url.rstrip("/"))
        return url.replace(scheme=base.scheme, hostname=base.hostname, port=base.port)

    scheme = _forwarded_scheme(request) or url.scheme
    hostname, port = _forwarded_host_and_port(request, scheme)

    replacements: dict[str, str | int | None] = {"scheme": scheme}
    if hostname:
        replacements["hostname"] = hostname
        replacements["port"] = port
    return url.replace(**replacements)


def _forwarded_scheme(request: Request) -> str | None:
    for header_name in ("x-forwarded-proto", "x-forwarded-scheme"):
        value = _first_header_value(request.headers.get(header_name))
        if value:
            return value

    forwarded = request.headers.get("forwarded")
    if forwarded:
        for item in forwarded.split(";"):
            key, _, value = item.partition("=")
            if key.strip().lower() == "proto":
                cleaned = value.strip().strip('"')
                if cleaned:
                    return cleaned

    forwarded_port = _first_header_value(request.headers.get("x-forwarded-port"))
    if forwarded_port == "443":
        return "https"
    if forwarded_port == "80":
        return "http"
    return None


def _forwarded_host_and_port(request: Request, scheme: str) -> tuple[str | None, int | None]:
    host_value = (
        _first_header_value(request.headers.get("x-forwarded-host"))
        or _forwarded_host_from_forwarded_header(request)
        or request.headers.get("host")
    )
    if not host_value:
        return None, None

    parsed = URL(f"{scheme}://{host_value}")
    return parsed.hostname, parsed.port


def _forwarded_host_from_forwarded_header(request: Request) -> str | None:
    forwarded = request.headers.get("forwarded")
    if not forwarded:
        return None
    for item in forwarded.split(";"):
        key, _, value = item.partition("=")
        if key.strip().lower() == "host":
            cleaned = value.strip().strip('"')
            if cleaned:
                return cleaned
    return None


def _first_header_value(value: str | None) -> str | None:
    if value is None:
        return None
    first = value.split(",", 1)[0].strip()
    return first or None
