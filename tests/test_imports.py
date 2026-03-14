from scbench_posttrain import TaskRegistry


def test_registry_smoke():
    registry = TaskRegistry()
    assert registry.names() == []
