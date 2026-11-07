"""The URL decision, blind-written against WP-docugraph-0.2.1 §3.1, §4, §5 and A-2.

Every DNS lookup is a fake resolver, so nothing here touches the network. A row
that says "resolver must not be called" uses a resolver that raises
``AssertionError``: if the decision reaches the network effect before rejecting
the value, the row fails instead of quietly passing on a cached answer.
"""

import pytest

from docugraph.security.url_policy import (
    RefusalReason,
    UrlKind,
    UrlPolicy,
    UrlRefusal,
    UrlRefusedError,
    format_refusal,
    resolve_host,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _never(host: str):
    """A resolver that fails the test if policy asks it anything."""
    raise AssertionError(f"resolver must not be called (host={host!r})")


def _mapping(table: dict[str, tuple[str, ...]]):
    """A dict-backed resolver; an unmapped host is a test-authoring error."""

    def resolve(host: str) -> tuple[str, ...]:
        try:
            return table[host]
        except KeyError:
            raise AssertionError(f"unexpected resolver lookup for {host!r}") from None

    return resolve


def _policy(kind: UrlKind = UrlKind.WEB, **kwargs) -> UrlPolicy:
    kwargs.setdefault("resolver", _never)
    return UrlPolicy(kind=kind, **kwargs)


def _refusal(policy: UrlPolicy, url: str) -> UrlRefusal:
    refusal = policy.check(url)
    assert refusal is not None, f"expected {url!r} to be refused"
    return refusal


def _assert_allowed(policy: UrlPolicy, url: str) -> None:
    refusal = policy.check(url)
    assert refusal is None, f"expected {url!r} to be allowed, got {refusal!r}"


# ---------------------------------------------------------------------------
# W1–W14: the worked examples
# ---------------------------------------------------------------------------


def test_w1_file_scheme_is_refused_without_dns() -> None:
    """ORACLE: W1 / D-1 — Crawl4AI accepts file://, so scheme is the gate."""
    refusal = _refusal(_policy(), "file:///etc/hosts")
    assert refusal.reason is RefusalReason.SCHEME_NOT_ALLOWED


def test_w2_raw_scheme_is_refused_without_dns() -> None:
    """ORACLE: W2 / D-1 — Crawl4AI's raw: prefix carries inline HTML."""
    refusal = _refusal(_policy(), "raw:<html>secret</html>")
    assert refusal.reason is RefusalReason.SCHEME_NOT_ALLOWED


def test_w3_loopback_ipv4_is_private() -> None:
    """ORACLE: W3 / §3.1 rule 5 — RFC 6890 via ipaddress.is_global."""
    policy = _policy(resolver=_mapping({"127.0.0.1": ("127.0.0.1",)}))
    refusal = _refusal(policy, "http://127.0.0.1:8080/")
    assert refusal.reason is RefusalReason.PRIVATE_ADDRESS


def test_w4_loopback_ipv6_is_private() -> None:
    """ORACLE: W4 / §3.1 rule 5."""
    policy = _policy(resolver=_mapping({"::1": ("::1",)}))
    refusal = _refusal(policy, "http://[::1]/")
    assert refusal.reason is RefusalReason.PRIVATE_ADDRESS


def test_w5_link_local_metadata_address_is_private() -> None:
    """ORACLE: W5 / §3.1 rule 5 — 169.254.169.254 is link-local."""
    policy = _policy(resolver=_mapping({"169.254.169.254": ("169.254.169.254",)}))
    refusal = _refusal(policy, "http://169.254.169.254/latest/meta-data/")
    assert refusal.reason is RefusalReason.PRIVATE_ADDRESS


def test_w6_rfc1918_address_is_private() -> None:
    """ORACLE: W6 / §3.1 rule 5 — RFC 1918."""
    policy = _policy(resolver=_mapping({"docs.internal.example": ("10.0.0.5",)}))
    refusal = _refusal(policy, "http://docs.internal.example/")
    assert refusal.reason is RefusalReason.PRIVATE_ADDRESS


def test_w7_one_private_address_refuses_the_host() -> None:
    """ORACLE: W7 / §3.1 rule 5 — every returned address must be global."""
    policy = _policy(
        resolver=_mapping({"cdn.example": ("93.184.216.34", "192.168.1.10")})
    )
    refusal = _refusal(policy, "https://cdn.example/")
    assert refusal.reason is RefusalReason.PRIVATE_ADDRESS


def test_w8_resolver_failure_fails_closed() -> None:
    """ORACLE: W8 / §1 — an unresolvable host is refused, not allowed."""

    def _raises(host: str):
        raise OSError(f"name resolution failed for {host}")

    refusal = _refusal(_policy(resolver=_raises), "https://nowhere.invalid/")
    assert refusal.reason is RefusalReason.UNRESOLVABLE


def test_w9_public_address_is_allowed() -> None:
    """ORACLE: W9 / D-1 — a global address under the defaults is allowed."""
    policy = _policy(resolver=_mapping({"docs.python.org": ("151.101.0.223",)}))
    _assert_allowed(policy, "https://docs.python.org/3/")


def test_w10_allow_private_permits_loopback() -> None:
    """ORACLE: W10 / D-1 — the intranet setting re-enables private addresses."""
    policy = _policy(
        allow_private=True, resolver=_mapping({"127.0.0.1": ("127.0.0.1",)})
    )
    _assert_allowed(policy, "http://127.0.0.1/")


def test_w11_allow_private_never_re_enables_a_refused_scheme() -> None:
    """ORACLE: W11 / §3.1 rule 2."""
    refusal = _refusal(_policy(allow_private=True), "file:///etc/hosts")
    assert refusal.reason is RefusalReason.SCHEME_NOT_ALLOWED


def test_w12_domain_outside_allowlist_is_refused_without_dns() -> None:
    """ORACLE: W12 / §3.3 — the allowlist is checked before the resolver."""
    policy = _policy(allowed_domains=("python.org",))
    refusal = _refusal(policy, "https://evil.example/")
    assert refusal.reason is RefusalReason.DOMAIN_NOT_ALLOWED


def test_w13_subdomain_of_an_allowed_domain_is_allowed() -> None:
    """ORACLE: W13 / §3.3 — docs.python.org is under python.org."""
    policy = _policy(
        allowed_domains=("python.org",),
        resolver=_mapping({"docs.python.org": ("151.101.0.223",)}),
    )
    _assert_allowed(policy, "https://docs.python.org/")


@pytest.mark.parametrize("url", ["http://0x7f000001/", "http://2130706433/"])
def test_w14_numeric_spellings_are_judged_by_address_not_string(url: str) -> None:
    """ORACLE: W14 / §3.1 rules 4–5 — numeric hosts are never allowed here.

    The system resolver is real but the host is numeric; no network traffic
    leaves the machine. A platform that rejects the spelling must report
    `unresolvable`; it must never fall through to allowed.
    """
    refusal = UrlPolicy(kind=UrlKind.WEB, resolver=resolve_host).check(url)
    assert refusal is not None, f"{url!r} resolved to a global address and was allowed"
    assert refusal.reason in {
        RefusalReason.PRIVATE_ADDRESS,
        RefusalReason.UNRESOLVABLE,
    }


# ---------------------------------------------------------------------------
# G1–G7: git-remote kind
# ---------------------------------------------------------------------------


def test_g1_option_like_remote_is_refused() -> None:
    """ORACLE: G1 / D-1 — git-clone(1): a leading - is an option."""
    refusal = _refusal(_policy(UrlKind.GIT_REMOTE), "--upload-pack=touch /tmp/x")
    assert refusal.reason is RefusalReason.OPTION_LIKE


def test_g2_short_option_like_remote_is_refused() -> None:
    """ORACLE: G2 / D-1."""
    refusal = _refusal(_policy(UrlKind.GIT_REMOTE), "-u touch /tmp/x")
    assert refusal.reason is RefusalReason.OPTION_LIKE


@pytest.mark.parametrize(
    "value", ["/Users/me/private-repo", "./repo", "file:///tmp/repo"]
)
def test_g3_local_paths_are_not_git_remotes(value: str) -> None:
    """ORACLE: G3 / D-1 — local paths and file:// are refused."""
    refusal = _refusal(_policy(UrlKind.GIT_REMOTE), value)
    assert refusal.reason is RefusalReason.SCHEME_NOT_ALLOWED


@pytest.mark.parametrize("value", ["ext::sh -c id", "fd::0"])
def test_g4_transport_helpers_are_refused(value: str) -> None:
    """ORACLE: G4 / git-remote-ext(1), git-remote-fd(1)."""
    refusal = _refusal(_policy(UrlKind.GIT_REMOTE), value)
    assert refusal.reason is RefusalReason.SCHEME_NOT_ALLOWED


def test_g5_private_https_remote_is_refused() -> None:
    """ORACLE: G5 / §3.1 rule 5."""
    policy = _policy(UrlKind.GIT_REMOTE, resolver=_mapping({"10.0.0.5": ("10.0.0.5",)}))
    refusal = _refusal(policy, "https://10.0.0.5/team/repo.git")
    assert refusal.reason is RefusalReason.PRIVATE_ADDRESS


def test_g6_public_https_remote_is_allowed() -> None:
    """ORACLE: G6 / D-1."""
    policy = _policy(
        UrlKind.GIT_REMOTE,
        resolver=_mapping({"github.com": ("140.82.112.3",)}),
    )
    _assert_allowed(policy, "https://github.com/CJ-coding-apps/docugraph.git")


def test_g7_scp_style_ssh_remote_is_allowed() -> None:
    """ORACLE: G7 / D-1 — scp-style ssh is a git remote, not a local path."""
    policy = _policy(
        UrlKind.GIT_REMOTE,
        resolver=_mapping({"github.com": ("140.82.112.3",)}),
    )
    _assert_allowed(policy, "git@github.com:CJ-coding-apps/docugraph.git")


# ---------------------------------------------------------------------------
# §5 edge cases
# ---------------------------------------------------------------------------


def test_idn_host_is_judged_by_its_resolved_address() -> None:
    """ORACLE: §5 — an IDN host is a host like any other.

    The resolver is given the Unicode host, or its IDNA form depending on the
    implementation; both spellings map to one public address, and the decision
    is the address's.
    """
    policy = _policy(
        resolver=_mapping(
            {
                "bücher.example": ("93.184.216.34",),
                "xn--bcher-kva.example": ("93.184.216.34",),
            }
        )
    )
    _assert_allowed(policy, "https://bücher.example/")


def test_trailing_dot_loopback_host_is_private() -> None:
    """ORACLE: §5 — a trailing dot does not change the address denoted."""
    policy = _policy(
        resolver=_mapping(
            {"localhost.": ("127.0.0.1",), "localhost": ("127.0.0.1",)}
        )
    )
    refusal = _refusal(policy, "http://localhost./")
    assert refusal.reason is RefusalReason.PRIVATE_ADDRESS


def test_uppercase_file_scheme_is_refused() -> None:
    """ORACLE: §5 — scheme comparison is case-insensitive."""
    refusal = _refusal(_policy(), "FILE:///etc/hosts")
    assert refusal.reason is RefusalReason.SCHEME_NOT_ALLOWED


def test_uppercase_http_scheme_is_still_judged() -> None:
    """ORACLE: §5 — HTTP is the same scheme as http."""
    policy = _policy(resolver=_mapping({"127.0.0.1": ("127.0.0.1",)}))
    refusal = _refusal(policy, "HTTP://127.0.0.1")
    assert refusal.reason is RefusalReason.PRIVATE_ADDRESS


def test_credentials_do_not_change_the_judged_host() -> None:
    """ORACLE: §5 — userinfo is not the host."""
    policy = _policy(resolver=_mapping({"127.0.0.1": ("127.0.0.1",)}))
    refusal = _refusal(policy, "http://user:pw@127.0.0.1/")
    assert refusal.reason is RefusalReason.PRIVATE_ADDRESS


def test_ipv4_mapped_ipv6_loopback_is_private() -> None:
    """ORACLE: §5 / A-2 — judged by the mapped IPv4."""
    policy = _policy(
        resolver=_mapping({"::ffff:127.0.0.1": ("::ffff:127.0.0.1",)})
    )
    refusal = _refusal(policy, "http://[::ffff:127.0.0.1]/")
    assert refusal.reason is RefusalReason.PRIVATE_ADDRESS


@pytest.mark.parametrize("url", ["", "   "])
def test_empty_or_whitespace_url_is_malformed(url: str) -> None:
    """ORACLE: §3.1 rule 1 / §5 — empty and whitespace-only are malformed."""
    refusal = _refusal(_policy(), url)
    assert refusal.reason is RefusalReason.MALFORMED


# ---------------------------------------------------------------------------
# A-2 additions
# ---------------------------------------------------------------------------


def test_unspecified_ipv4_is_private() -> None:
    """ORACLE: A-2 — 0.0.0.0 is not a global address."""
    policy = _policy(resolver=_mapping({"0.0.0.0": ("0.0.0.0",)}))
    refusal = _refusal(policy, "http://0.0.0.0/")
    assert refusal.reason is RefusalReason.PRIVATE_ADDRESS


def test_parser_differential_backslash_authority_is_malformed() -> None:
    """ORACLE: A-2 — must not be judged as the host urlsplit names (evil.example)."""
    refusal = _refusal(_policy(), "http://127.0.0.1\\@evil.example/")
    assert refusal.reason is RefusalReason.MALFORMED


@pytest.mark.parametrize(
    "url",
    ["http://example.com/\x00", "http://exa\tmple.com/", "http://example.com/\r"],
)
def test_control_characters_are_malformed(url: str) -> None:
    """ORACLE: A-2 — any ASCII control character in an allowed-scheme value."""
    refusal = _refusal(_policy(), url)
    assert refusal.reason is RefusalReason.MALFORMED


def test_whitespace_in_path_is_allowed() -> None:
    """ORACLE: A-2 — only the raw authority is character-screened."""
    policy = _policy(resolver=_mapping({"example.com": ("93.184.216.34",)}))
    _assert_allowed(policy, "https://example.com/a b")


def test_whitespace_in_raw_authority_is_malformed() -> None:
    """ORACLE: A-2 — a space in the raw authority is refused."""
    refusal = _refusal(_policy(), "https://exa mple.com/")
    assert refusal.reason is RefusalReason.MALFORMED


def test_scoped_ipv6_literal_is_malformed_even_when_private_is_allowed() -> None:
    """ORACLE: A-2 / A-10 — scoped IPv6 is malformed, whatever allow_private says."""
    policy = _policy(
        allow_private=True,
        resolver=_mapping({"fe80::1%25eth0": ("fe80::1%25eth0",)}),
    )
    refusal = _refusal(policy, "http://[fe80::1%25eth0]/")
    assert refusal.reason is RefusalReason.MALFORMED
    assert "scoped IPv6" in refusal.detail


def test_ipv4_mapped_public_address_is_allowed() -> None:
    """ORACLE: A-2 — IPv4-mapped IPv6 is judged by its mapped IPv4 (global)."""
    policy = _policy(
        resolver=_mapping({"::ffff:9.9.9.9": ("::ffff:9.9.9.9",)})
    )
    _assert_allowed(policy, "http://[::ffff:9.9.9.9]/")


def test_out_of_range_port_is_malformed() -> None:
    """ORACLE: A-2 — urlsplit(...).port raising ValueError is malformed."""
    refusal = _refusal(_policy(), "http://example.com:99999/")
    assert refusal.reason is RefusalReason.MALFORMED


# ---------------------------------------------------------------------------
# A-9: DNS memoization on the policy instance
# ---------------------------------------------------------------------------


def test_a_host_is_resolved_once_for_repeated_checks() -> None:
    """ORACLE: A-9 — successes are memoized per host for the crawl."""
    calls: list[str] = []

    def resolve(host: str) -> tuple[str, ...]:
        calls.append(host)
        return ("1.1.1.1",)

    policy = _policy(resolver=resolve)
    _assert_allowed(policy, "https://example.com/one")
    _assert_allowed(policy, "https://example.com/two")
    assert calls == ["example.com"]


def test_a_resolver_failure_is_memoized_too() -> None:
    """ORACLE: A-9 — OSError is cached; the second check does not re-query."""
    calls: list[str] = []

    def resolve(host: str) -> tuple[str, ...]:
        calls.append(host)
        raise OSError("name resolution failed")

    policy = _policy(resolver=resolve)
    first = _refusal(policy, "https://nowhere.invalid/")
    second = _refusal(policy, "https://nowhere.invalid/")
    assert first.reason is RefusalReason.UNRESOLVABLE
    assert second.reason is RefusalReason.UNRESOLVABLE
    assert calls == ["nowhere.invalid"]


# ---------------------------------------------------------------------------
# The shell bridge strings
# ---------------------------------------------------------------------------


def test_format_refusal_is_exact() -> None:
    """ORACLE: A-10 / §3.4 — `Refused: <reason> — <detail>` on one line."""
    refusal = UrlRefusal(RefusalReason.PRIVATE_ADDRESS, "x")
    assert format_refusal(refusal) == "Refused: private_address — x"


def test_url_refused_error_carries_the_refusal() -> None:
    """ORACLE: §3.1 — UrlRefusedError preserves the decision it was built from."""
    refusal = UrlRefusal(RefusalReason.SCHEME_NOT_ALLOWED, "file scheme")
    error = UrlRefusedError(refusal)
    assert error.refusal is refusal


# ---------------------------------------------------------------------------
# A-2 (0.3.0): link-local and cloud-metadata addresses are never allowed
# ---------------------------------------------------------------------------


def test_p1_metadata_ipv4_is_refused_even_when_private_is_allowed() -> None:
    """ORACLE: A-2 / P1 — 169.254.169.254 is metadata, not merely private."""
    policy = _policy(
        allow_private=True,
        resolver=_mapping({"169.254.169.254": ("169.254.169.254",)}),
    )
    refusal = _refusal(policy, "http://169.254.169.254/latest/meta-data/")
    assert refusal.reason is RefusalReason.PRIVATE_ADDRESS
    assert "never allowed" in refusal.detail


def test_p2_aws_ipv6_metadata_is_refused_even_when_private_is_allowed() -> None:
    """ORACLE: A-2 / P2 — fd00:ec2::254 is the AWS IPv6 metadata endpoint."""
    policy = _policy(
        allow_private=True,
        resolver=_mapping({"fd00:ec2::254": ("fd00:ec2::254",)}),
    )
    refusal = _refusal(policy, "http://[fd00:ec2::254]/")
    assert refusal.reason is RefusalReason.PRIVATE_ADDRESS


def test_p3_alibaba_metadata_is_refused_even_when_private_is_allowed() -> None:
    """ORACLE: A-2 / P3 — 100.100.100.200 is a cloud-metadata address."""
    policy = _policy(
        allow_private=True,
        resolver=_mapping({"metadata.example": ("100.100.100.200",)}),
    )
    refusal = _refusal(policy, "http://metadata.example/")
    assert refusal.reason is RefusalReason.PRIVATE_ADDRESS


def test_p4_mapped_metadata_is_refused_even_when_private_is_allowed() -> None:
    """ORACLE: A-2 / P4 — an IPv4-mapped literal is judged by its IPv4."""
    policy = _policy(
        allow_private=True,
        resolver=_mapping({"::ffff:169.254.169.254": ("::ffff:169.254.169.254",)}),
    )
    refusal = _refusal(policy, "http://[::ffff:169.254.169.254]/")
    assert refusal.reason is RefusalReason.PRIVATE_ADDRESS
    assert "169.254.169.254" in refusal.detail
    assert "never allowed" in refusal.detail


def test_p5_link_local_ipv6_is_refused_even_when_private_is_allowed() -> None:
    """ORACLE: A-2 / P5 — fe80::/10 is link-local and never allowed."""
    policy = _policy(
        allow_private=True,
        resolver=_mapping({"fe80::1": ("fe80::1",)}),
    )
    refusal = _refusal(policy, "http://[fe80::1]/")
    assert refusal.reason is RefusalReason.PRIVATE_ADDRESS


def test_p6_allow_private_still_admits_an_ordinary_private_address() -> None:
    """ORACLE: A-2 / P6 — the intranet setting still admits RFC 1918."""
    policy = _policy(
        allow_private=True,
        resolver=_mapping({"docs.internal": ("10.0.0.5",)}),
    )
    _assert_allowed(policy, "http://docs.internal/")


def test_p7_metadata_is_refused_under_the_default_setting() -> None:
    """ORACLE: A-2 / P7 — the default policy refuses metadata too."""
    policy = _policy(resolver=_mapping({"169.254.169.254": ("169.254.169.254",)}))
    refusal = _refusal(policy, "http://169.254.169.254/")
    assert refusal.reason is RefusalReason.PRIVATE_ADDRESS


def test_p9_ipv4_compatible_metadata_is_refused() -> None:
    """ORACLE: A-2 / P9 — ::a9fe:a9fe carries 169.254.169.254."""
    policy = _policy(resolver=_mapping({"::a9fe:a9fe": ("::a9fe:a9fe",)}))
    refusal = _refusal(policy, "http://[::a9fe:a9fe]/")
    assert refusal.reason is RefusalReason.PRIVATE_ADDRESS
    assert "169.254.169.254" in refusal.detail
    assert "never allowed" in refusal.detail


def test_p10_nat64_metadata_and_private_are_refused() -> None:
    """ORACLE: A-2 / P10 — NAT64-embedded IPv4 is judged by the embedded address."""
    policy = _policy(
        resolver=_mapping(
            {
                "64:ff9b::a9fe:a9fe": ("64:ff9b::a9fe:a9fe",),
                "64:ff9b::10.0.0.5": ("64:ff9b::10.0.0.5",),
            }
        )
    )

    metadata = _refusal(policy, "http://[64:ff9b::a9fe:a9fe]/")
    assert metadata.reason is RefusalReason.PRIVATE_ADDRESS
    assert "169.254.169.254" in metadata.detail

    private = _refusal(policy, "http://[64:ff9b::10.0.0.5]/")
    assert private.reason is RefusalReason.PRIVATE_ADDRESS
    assert "10.0.0.5" in private.detail


def test_p11_nat64_64_ff9b_1_metadata_is_refused_even_when_private_is_allowed() -> None:
    """ORACLE: A-2 / P11 — RFC 6052 prefix 64:ff9b:1::/48 embeds IPv4 in bits 48-79."""
    policy = _policy(
        allow_private=True,
        resolver=_mapping({"64:ff9b:1:a9fe:a9fe::": ("64:ff9b:1:a9fe:a9fe::",)}),
    )
    refusal = _refusal(policy, "http://[64:ff9b:1:a9fe:a9fe::]/")
    assert refusal.reason is RefusalReason.PRIVATE_ADDRESS
    assert "169.254.169.254" in refusal.detail


def test_p12_6to4_metadata_is_refused_even_when_private_is_allowed() -> None:
    """ORACLE: A-2 / P12 — a 6to4 literal is judged by its embedded IPv4."""
    policy = _policy(
        allow_private=True,
        resolver=_mapping({"2002:a9fe:a9fe::": ("2002:a9fe:a9fe::",)}),
    )
    refusal = _refusal(policy, "http://[2002:a9fe:a9fe::]/")
    assert refusal.reason is RefusalReason.PRIVATE_ADDRESS


def test_p13_teredo_server_and_client_metadata_are_both_judged() -> None:
    """ORACLE: A-2 / P13 — both the Teredo client and server IPv4 are judged."""
    policy = _policy(
        allow_private=True,
        resolver=_mapping(
            {
                "2001:0:4136:e378:8000:63bf:5601:5601": (
                    "2001:0:4136:e378:8000:63bf:5601:5601",
                ),
                "2001:0:a9fe:a9fe:8000:63bf:0000:0001": (
                    "2001:0:a9fe:a9fe:8000:63bf:0000:0001",
                ),
            }
        ),
    )

    client = _refusal(policy, "http://[2001:0:4136:e378:8000:63bf:5601:5601]/")
    assert client.reason is RefusalReason.PRIVATE_ADDRESS
    assert "169.254.169.254" in client.detail

    server = _refusal(policy, "http://[2001:0:a9fe:a9fe:8000:63bf:0000:0001]/")
    assert server.reason is RefusalReason.PRIVATE_ADDRESS
    assert "169.254.169.254" in server.detail
