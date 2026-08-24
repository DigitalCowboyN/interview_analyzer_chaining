from tools.graph.flow import handler_labels


def test_handler_labels_from_merge_matching_glossary():
    # SentenceCreatedHandler writes MERGE (s:Fragment ...); Fragment IS a glossary term
    labels = handler_labels("projections.handlers.sentence_handlers.SentenceCreatedHandler", ".")
    assert "Fragment" in labels
