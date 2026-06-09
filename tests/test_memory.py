"""Tests for PromptMemory adaptive feedback loop."""
from voiceprint.memory import PromptMemory


class TestPromptMemory:
    def test_initial_state(self):
        m = PromptMemory()
        assert m.total_runs() == 0
        assert m.counts() == {}
        assert m.best_level(default=2) == 2

    def test_record_and_count(self):
        m = PromptMemory()
        m.record(0, 0.1)
        m.record(0, 0.2)
        m.record(1, 0.9)
        assert m.total_runs() == 3
        assert m.counts() == {0: 2, 1: 1}

    def test_success_rate(self):
        m = PromptMemory()
        m.record(0, 0.1)
        m.record(0, 0.6)
        m.record(0, 0.3)
        assert m.success_rate(0) == 2 / 3
        assert m.success_rate(9) == 0.0

    def test_best_level_picks_lowest_avg(self):
        m = PromptMemory()
        m.record(0, 0.9)  # level 0 avg = 0.9
        m.record(1, 0.1)  # level 1 avg = 0.1
        m.record(2, 0.9)
        m.record(2, 0.5)  # level 2 avg = 0.7
        assert m.best_level(default=0) == 1

    def test_best_level_default_when_no_data(self):
        m = PromptMemory()
        assert m.best_level(default=3) == 3

    def test_reset(self):
        m = PromptMemory()
        m.record(0, 0.1)
        m.reset()
        assert m.total_runs() == 0
        assert m.best_level(default=99) == 99

    def test_thread_safety(self):
        import threading
        m = PromptMemory()
        errors = []

        def record_many(level):
            for _ in range(100):
                try:
                    m.record(level, 0.5)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=record_many, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert m.total_runs() == 400
