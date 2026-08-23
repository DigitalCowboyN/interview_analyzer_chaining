from tools.graph.flow import register_map


def test_register_map_bridges_type_to_data_class_and_handler():
    m = register_map(".")
    # registry.register("InterviewCreated", InterviewCreatedHandler(...)) in projections.bootstrap
    assert m.get("events.interview_events.InterviewCreatedData") == \
        "projections.handlers.interview_handlers.InterviewCreatedHandler"
