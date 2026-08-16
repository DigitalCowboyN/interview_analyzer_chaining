from tools.graph.traverse import _entry_addresses


def test_type_selector_returns_all_of_a_type():
    addrs = _entry_addresses("type:Capability", root=".")
    assert addrs and all(a.startswith("capabilities:") for a in addrs)


def test_under_selector_returns_code_units_below_path():
    addrs = _entry_addresses("under:src/api/", root=".")
    assert all(a.startswith("code:") for a in addrs)


def test_plain_address_passes_through():
    assert _entry_addresses("code:api", root=".") == ["code:api"]
