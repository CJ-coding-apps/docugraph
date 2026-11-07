"""The URL decision for crawled pages and git remotes.

One public entry point, :meth:`UrlPolicy.check`, returns ``None`` for an
allowed value and a :class:`UrlRefusal` otherwise. The decision is a function
of the parsed value and the policy; the DNS lookup is an injected effect, so
every rule can be exercised without a network. A lookup is memoized per host
for the life of the policy instance, which the MCP layer builds once per tool
call (one crawl).
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from urllib.parse import urlsplit


class UrlKind(StrEnum):
    """What the URL is used for; each kind has its own scheme allowlist."""

    WEB = "web"
    GIT_REMOTE = "git-remote"


class RefusalReason(StrEnum):
    """Why a value was refused. Closed vocabulary; one reason per decision."""

    MALFORMED = "malformed"
    SCHEME_NOT_ALLOWED = "scheme_not_allowed"
    OPTION_LIKE = "option_like"
    DOMAIN_NOT_ALLOWED = "domain_not_allowed"
    UNRESOLVABLE = "unresolvable"
    PRIVATE_ADDRESS = "private_address"
    PATH_ESCAPE = "path_escape"
    UNSUPPORTED_CRAWLER = "unsupported_crawler"


@dataclass(frozen=True, slots=True)
class UrlRefusal:
    """A refusal reason and one line of detail for the human reading it."""

    reason: RefusalReason
    detail: str


Resolver = Callable[[str], Sequence[str]]

_WEB_SCHEMES = frozenset({"http", "https"})
_GIT_SCHEMES = frozenset({"https", "ssh"})
_PASS_THROUGH_SCHEMES = frozenset({"data", "blob", "about"})

# Link-local and cloud-metadata endpoints. They stay refused whatever
# ``allow_private`` says: "this intranet" never means "this machine's cloud
# credentials". 169.254.0.0/16 and fe80::/10 are link-local; the other two are
# the AWS and Alibaba metadata services.
_METADATA_NETWORKS = (
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("fd00:ec2::254/128"),
    ipaddress.ip_network("100.100.100.200/32"),
)

# Address formats that carry an IPv4 inside an IPv6 literal. ``is_global`` on
# the IPv6 wrapper is not the IPv4's answer: 64:ff9b::10.0.0.5 is reported
# global while denoting a private address.
_IPV4_COMPATIBLE = ipaddress.ip_network("::/96")
_NAT64_96 = ipaddress.ip_network("64:ff9b::/96")
# RFC 6052 puts the IPv4 in bits 48-79 for a /48 prefix, not in the low 32.
_NAT64_48 = ipaddress.ip_network("64:ff9b:1::/48")


def resolve_host(host: str) -> tuple[str, ...]:
    """Resolve ``host`` with the system resolver; raises ``OSError`` on failure.

    Only the addresses matter, so the port and socket type are fixed and
    duplicates are removed in order.
    """
    infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    addresses: list[str] = []
    for info in infos:
        address = info[4][0]
        if not isinstance(address, str):
            continue
        if address not in addresses:
            addresses.append(address)
    return tuple(addresses)


def format_refusal(refusal: UrlRefusal) -> str:
    """The one-line form the MCP surface returns: ``Refused: <reason> — <detail>``."""
    return f"Refused: {refusal.reason} — {refusal.detail}"


class UrlRefusedError(RuntimeError):
    """A shell bridge that carries the decision it was raised from."""

    def __init__(self, refusal: UrlRefusal) -> None:
        self.refusal = refusal
        super().__init__(format_refusal(refusal))


def _refuse(reason: RefusalReason, detail: str) -> UrlRefusal:
    return UrlRefusal(reason=reason, detail=detail)


def _has_control_character(value: str) -> bool:
    return any(ord(char) < 0x20 or ord(char) == 0x7F for char in value)


def _authority_is_suspicious(authority: str) -> bool:
    """A backslash or any whitespace in the raw authority is refused.

    The raw authority is used deliberately: ``http://127.0.0.1\\@evil.example/``
    parses its hostname as ``evil.example``, which is exactly the value a
    hostname-only check would wrongly trust.
    """
    return "\\" in authority or any(char.isspace() for char in authority)


def _normalize_domain(domain: str) -> str:
    return domain.strip().lower().rstrip(".")


def _domain_allowed(host: str, allowed_domains: tuple[str, ...]) -> bool:
    if not allowed_domains:
        return True

    normalized_host = host.lower().rstrip(".")
    try:
        ipaddress.ip_address(normalized_host)
    except ValueError:
        is_ip_literal = False
    else:
        is_ip_literal = True

    for domain in allowed_domains:
        normalized_domain = _normalize_domain(domain)
        if not normalized_domain:
            continue
        if is_ip_literal:
            # An address is matched exactly; a suffix match on digits would
            # make 127.0.0.10 "inside" 127.0.0.1.
            if normalized_host == normalized_domain:
                return True
        elif normalized_host == normalized_domain or normalized_host.endswith(
            "." + normalized_domain
        ):
            return True
    return False


def _is_metadata_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """True for link-local and cloud-metadata endpoints."""
    return any(address in network for network in _METADATA_NETWORKS)


def _embedded_ipv4s(address: ipaddress.IPv6Address) -> tuple[ipaddress.IPv4Address, ...]:
    """Every IPv4 an IPv6 literal denotes, in the order the forms are checked.

    These forms exist so an IPv6 spelling can carry an IPv4 past a check that
    only looks at the wrapper: IPv4-mapped, IPv4-compatible, NAT64 (both
    prefixes), 6to4 and Teredo (its server *and* its client). Duplicates are
    removed in order.
    """
    candidates: list[ipaddress.IPv4Address] = []

    mapped = address.ipv4_mapped
    if mapped is not None:
        candidates.append(mapped)

    value = int(address)
    if address in _IPV4_COMPATIBLE or address in _NAT64_96:
        candidates.append(ipaddress.IPv4Address(value & 0xFFFFFFFF))
    if address in _NAT64_48:
        candidates.append(ipaddress.IPv4Address((value >> 48) & 0xFFFFFFFF))

    sixtofour = address.sixtofour
    if sixtofour is not None:
        candidates.append(sixtofour)

    teredo = address.teredo
    if teredo is not None:
        candidates.extend(teredo)

    deduped: list[ipaddress.IPv4Address] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return tuple(deduped)


def _judge_addresses(addresses: Sequence[str], allow_private: bool) -> UrlRefusal | None:
    for raw_address in addresses:
        if "%" in raw_address:
            return _refuse(RefusalReason.MALFORMED, "scoped IPv6 literal")

        try:
            address = ipaddress.ip_address(raw_address)
        except ValueError:
            return _refuse(
                RefusalReason.UNRESOLVABLE,
                f"resolver returned an unparsable address {raw_address!r}",
            )

        embedded: tuple[ipaddress.IPv4Address, ...] = ()
        if isinstance(address, ipaddress.IPv6Address):
            embedded = _embedded_ipv4s(address)

        # Link-local and cloud metadata are refused even when private
        # networks are allowed; the detail names the address that failed.
        for candidate in (address, *embedded):
            if _is_metadata_address(candidate):
                return _refuse(
                    RefusalReason.PRIVATE_ADDRESS,
                    f"{candidate} is a link-local or cloud-metadata address and is never allowed",
                )

        if not allow_private:
            for candidate in embedded:
                if not candidate.is_global:
                    return _refuse(
                        RefusalReason.PRIVATE_ADDRESS,
                        f"{candidate} is not a global address",
                    )
            if not address.is_global:
                return _refuse(
                    RefusalReason.PRIVATE_ADDRESS,
                    f"{address} is not a global address",
                )
    return None


@dataclass(frozen=True, slots=True)
class UrlPolicy:
    """A value-in, decision-out URL policy.

    ``resolver`` is the only effect; it is called once per distinct host for
    the life of this instance. ``allowed_domains`` is a host allowlist: an
    entry matches itself and its subdomains, an IP literal matches exactly.
    """

    kind: UrlKind
    resolver: Resolver = resolve_host
    allow_private: bool = False
    allowed_domains: tuple[str, ...] = ()
    _dns: dict[str, tuple[str, ...] | OSError] = field(
        default_factory=dict, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "allowed_domains",
            tuple(_normalize_domain(d) for d in self.allowed_domains if d.strip()),
        )

    def check(self, url: str) -> UrlRefusal | None:
        """Return ``None`` when ``url`` is allowed, otherwise the refusal."""
        raw = url if isinstance(url, str) else str(url)
        stripped = raw.strip()
        if not stripped:
            return _refuse(RefusalReason.MALFORMED, "empty URL")

        if stripped.startswith("-"):
            return _refuse(
                RefusalReason.OPTION_LIKE, "value begins with '-' and would be an option"
            )

        parts = urlsplit(stripped)

        scp_style = False
        if self.kind is UrlKind.GIT_REMOTE and not parts.scheme:
            scp_style = self._looks_like_scp_style(parts.path)

        scheme = parts.scheme.lower()
        if self.kind is UrlKind.WEB:
            if scheme not in _WEB_SCHEMES:
                return _refuse(
                    RefusalReason.SCHEME_NOT_ALLOWED, f"scheme {scheme!r} is not allowed"
                )
        elif scheme not in _GIT_SCHEMES and not scp_style:
            return _refuse(RefusalReason.SCHEME_NOT_ALLOWED, f"scheme {scheme!r} is not allowed")

        # A-2: character rules apply only after the scheme decision, so a
        # refused scheme keeps its scheme_not_allowed reason.
        if _has_control_character(raw):
            return _refuse(RefusalReason.MALFORMED, "ASCII control character in the URL")

        if scp_style:
            authority = parts.path.split(":", 1)[0]
            host = authority.rsplit("@", 1)[-1]
        else:
            authority = parts.netloc
            host = parts.hostname or ""

        if _authority_is_suspicious(authority):
            return _refuse(
                RefusalReason.MALFORMED,
                "backslash or whitespace in the raw authority",
            )

        try:
            parts.port  # noqa: B018 -- accessing it validates the port
        except ValueError:
            return _refuse(RefusalReason.MALFORMED, "port is not a valid integer")

        if not host:
            return _refuse(RefusalReason.MALFORMED, "URL has no host")

        if "%" in host:
            return _refuse(RefusalReason.MALFORMED, "scoped IPv6 literal")

        if not _domain_allowed(host, self.allowed_domains):
            return _refuse(
                RefusalReason.DOMAIN_NOT_ALLOWED,
                f"host {host!r} is not in allowed_domains",
            )

        addresses = self._resolve(host)
        if isinstance(addresses, OSError):
            return _refuse(RefusalReason.UNRESOLVABLE, f"could not resolve {host!r}: {addresses}")
        if not addresses:
            return _refuse(RefusalReason.UNRESOLVABLE, f"{host!r} resolved to no address")

        return _judge_addresses(addresses, self.allow_private)

    @staticmethod
    def _looks_like_scp_style(path: str) -> bool:
        """``user@host:path`` -- the ssh shorthand git accepts without a scheme."""
        if "@" not in path:
            return False
        authority, separator, remainder = path.partition(":")
        if not separator or not remainder:
            return False
        user, _, host = authority.rpartition("@")
        return bool(user) and bool(host) and "/" not in host and "\\" not in host

    def _resolve(self, host: str) -> tuple[str, ...] | OSError:
        """Resolve through the per-instance memo; failures are cached too."""
        key = host.lower().rstrip(".")
        if key in self._dns:
            return self._dns[key]
        try:
            addresses = tuple(self.resolver(key))
        except OSError as error:
            self._dns[key] = error
            return error
        self._dns[key] = addresses
        return addresses


def is_pass_through_scheme(scheme: str) -> bool:
    """Browser schemes that are not network requests (``data:``, ``blob:``, ``about:``)."""
    return scheme.lower() in _PASS_THROUGH_SCHEMES
