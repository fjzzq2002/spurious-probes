from spurious_probes.normalize import first_list_item, normalize

P = {"id": "x", "norm": "phrase", "text": "Name a type of life stage."}


def cat(raw):
    return normalize(P, first_list_item(raw))["cat"]


def test_list_after_leadin():
    assert cat("Here are some life stages:\n\n1. **Infancy**\n2. Childhood\n3. Adolescence") == "infancy"
    assert cat("# Life Stages\n\n- Infancy\n- Toddlerhood") == "infancy"


def test_list_without_leadin():
    assert cat("- Adolescence\n- Adulthood") == "adolescence"
    assert cat("- Infancy") == "infancy"
    assert cat("1) Infancy\n2) Childhood") == "infancy"


def test_non_list_untouched():
    for raw in ["Adolescence.", "**Adolescence**\n\nAdolescence is the transitional stage...", "Adolescence\n\nThis is the period...",
                "One type of life stage is adolescence."]:
        assert first_list_item(raw) == raw
        assert cat(raw) == "adolescence"


def test_prose_then_list_keeps_first_line():
    raw = "Adolescence is a good example.\n- Infancy\n- Childhood"
    assert first_list_item(raw) == raw
